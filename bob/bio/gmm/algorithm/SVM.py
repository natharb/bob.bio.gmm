#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>


import bob.core
import bob.io.base
import bob.learn.em

import numpy

from bob.bio.base.algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.gmm")

from .GMM import GMMRegular

class SVMGMM (GMMRegular):
  """
  Trains 1 vs All SVM using the GMM supervectors
  """

  def __init__(self, **kwargs):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""

#    logger.warn("This class must be checked. Please verify that I didn't do any mistake here. I had to rename 'train_projector' into a 'train_enroller'!")
    # initialize the UBMGMM base class
    GMMRegular.__init__(self, **kwargs)
    # register a different set of functions in the Tool base class
    Algorithm.__init__(self, requires_enroller_training = True, performs_projection = False)


  #######################################################
  ################ UBM training #########################

  def train_enroller(self, train_features, enroller_file, metadata=None):
    """Computes the Universal Background Model from the training ("world") data"""

    # stacking all the features. TODO: This is super sub-optimal
    train_features_flatten = numpy.vstack([feature for client in train_features for feature in client])

    # training UBM (it's on self.ubm)
    self.train_ubm(train_features_flatten)

    # Now it comes the hack.
    # We would need to stack the features from all classes

    # Setting the MAP Trainer
    self.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.ubm, relevance_factor = self.relevance_factor, update_means = True, update_variances = False)

    # Efficiency tip, let's pre-allocate the supervector arrays
    mean_supervectors = []
    for client in train_features:
        shape = (len(client), self.ubm.mean_supervector.shape[0])
        mean_supervectors.append(numpy.zeros(shape))

    # Now let's compute the supervectors
    for client,i in zip(train_features, range(len(train_features))):
        for feature,j in zip(client, range(len(client))):
            # Running MAP
            map_feature = self.enroll_gmm(feature)
            mean_supervectors[i][j] = map_feature.mean_supervector
    
    # The enroller is composed by the UBM and all the training supervector samples

    # saving ubm
    hdf5 = bob.io.base.HDF5File(enroller_file, "w")
    hdf5.create_group("/UBM")
    hdf5.cd("/UBM")
    self.ubm.save(hdf5)

    # saving supervectors
    hdf5.create_group("/train_supervectors")
    hdf5.cd("/train_supervectors")
    for i in range(len(mean_supervectors)):
        hdf5.set("{0}".format(i), mean_supervectors[i])
    

  ######################################################
  ################ Feature comparison ##################
  def read_model(self, model_file):
    """Reads the model, which is a GMM machine"""
    return bob.learn.em.GMMMachine(bob.io.base.HDF5File(model_file))

  def score(self, model, probe):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    assert isinstance(model, bob.learn.em.GMMMachine)
    assert isinstance(probe, bob.learn.em.GMMStats)
    return self.scoring_function([model], self.ubm, [probe], [], frame_length_normalisation = True)[0][0]

  def score_for_multiple_probes(self, model, probes):
    """This function computes the score between the given model and several given probe files."""
    assert isinstance(model, bob.learn.em.GMMMachine)
    for probe in probes:
      assert isinstance(probe, bob.learn.em.GMMStats)
#    logger.warn("Please verify that this function is correct")
    return self.probe_fusion_function(self.scoring_function([model], self.ubm, probes, [], frame_length_normalisation = True))


  #######################################################
  ############## GMM training using UBM #################

  def load_enroller(self, enroller_file):
    """Reads the UBM model from file"""

     # saving ubm
    hdf5 = bob.io.base.HDF5File(enroller_file)
    hdf5.cd("/UBM")
    self.ubm = bob.learn.em.GMMMachine(hdf5)

    # saving supervectors
    self.negative_samples = []
    hdf5.cd("/train_supervectors")
    for i in range(len(hdf5.keys())):
        self.negative_samples.append(hdf5.get("{0}".format(i)))
       
    return [self.ubm, self.negative_samples]


  def enroll(self, feature_arrays, metadata=None):
    """
    Enrolling SVM with super vectors
    """

    # Setting the MAP Trainer
    self.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.ubm, relevance_factor = self.relevance_factor, update_means = True, update_variances = False)

    # Efficiency tip, let's pre-allocate the supervector arrays   
    shape = (len(feature_arrays), self.ubm.mean_supervector.shape[0])
    mean_supervectors = numpy.zeros(shape)
   
    for feature, i in zip(feature_arrays, range(len(feature_arrays))):
      map_feature = self.enroll_gmm(feature)
      mean_supervectors[i] = map_feature.mean_supervector


    ############
    #  TODO: DO SVM here
    ############

    import ipdb; ipdb.set_trace()
    pass



  ######################################################
  ################ Feature comparison ##################
  def score(self, model, probe):
    """Computes the score for the given model and the given probe.
    The score are Log-Likelihood.
    Therefore, the log of the likelihood ratio is obtained by computing the following difference."""

    assert isinstance(model, bob.learn.em.GMMMachine)
    self._check_feature(probe)
    score = sum(model.log_likelihood(probe[i,:]) - self.ubm.log_likelihood(probe[i,:]) for i in range(probe.shape[0]))
    return score/probe.shape[0]

  def score_for_multiple_probes(self, model, probes):
    raise NotImplementedError("Implement Me!")
