#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


import bob.core
import bob.io.base
import bob.learn.em
from bob.bio.base.tools.FileSelector import FileSelector
import bob.learn.libsvm
import numpy

from bob.bio.base.algorithm import Algorithm

import logging
logger = logging.getLogger("bob.bio.gmm")

from .GMM import GMMRegular

class SVMGMM (GMMRegular):
  """
  Trains 1 vs All SVM using the GMM supervectors


  Parameters
  ----------

   machine_type: :py:class:`str`
      A type of the SVM machine. Please check ``bob.learn.libsvm`` for
      more details. Default: 'C_SVC'.

   kernel_type: :py:class:`str`
      A type of kerenel for the SVM machine. Please check ``bob.learn.libsvm``
      for more details. Default: 'RBF'.

   C: float
     Regularization value for the C SVM

   gamma: float
     Gamma parameter for the RBF Kernel


  """

  def __init__(self,           
            machine_type='C_SVC',
            kernel_type='RBF',
            C=1.,
            gamma=0.1,

          **kwargs):

    # initialize the UBMGMM base class
    GMMRegular.__init__(self, **kwargs)
    # register a different set of functions in the Tool base class
    Algorithm.__init__(self, requires_enroller_training = True, performs_projection = False)

    self.machine_type = machine_type
    self.kernel_type  = kernel_type
    self.C            = C
    self.gamma        = gamma


  #######################################################
  ################ UBM training #########################

  def train_enroller(self, train_features, enroller_file, metadata=None):
    """Computes the Universal Background Model from the training ("world") data"""    

    ######################################
    # TODO: This is a critical moment.
    # With the next two lines of code we are breaking completely the isolation concept implemented
    # in bob.bio.base by introducing database knowledge inside of the algorithm.
    # This is a total HACK.
    # In short, we just opened the gates from hell.
    # Some demons may come out and the might terrorize innocent people.
    # Do your prayers, you will need them.
    # Only faith can save your soul.
    # God forgive us
    fs = FileSelector.instance()
    #train_files = fs.training_objects('extracted', 'train_projector', arrange_by_client = True)
    train_files = fs.database.training_files('train_projector', True)

    #####

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
        # Fetching and memorizing the client id, so we can use it during the enroll
        class_id = train_files[i][0].client_id
        hdf5.set("{0}".format(class_id), mean_supervectors[i])
    

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
    self.negative_samples = dict()
    hdf5.cd("/train_supervectors")
    for i in hdf5.keys():
        self.negative_samples[i] = hdf5.get("{0}".format(i))
       
    #return [self.ubm, self.negative_samples]


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

    # THIS IS ANOTHER HACK
    # Stacking negative samples

    all_negative_samples = None
    class_id = metadata[0].client_id
   
    for k in self.negative_samples.keys():
        # If it is from the same class, skip it
        if class_id in k: 
            continue

        if all_negative_samples is None:
            all_negative_samples = self.negative_samples[k]
        else:
            all_negative_samples = numpy.vstack((all_negative_samples, self.negative_samples[k]))
  
   
    # NATH
    # YOU CAN IMPLEMENT SOME DATA NORMALIZATION HERE
    # SUPER RECOMMENDED

    # initialize the SVM trainer:
    trainer = bob.learn.libsvm.Trainer(machine_type=self.machine_type,
                                       kernel_type=self.kernel_type,
                                       probability=True)
    trainer.gamma = self.gamma
    trainer.cost = self.C
    machine = trainer.train([mean_supervectors,
                             all_negative_samples])

    return machine


  def read_model(self, model_file):
    """Reads the model, which is a SVM  machine"""
    
    f = bob.io.base.HDF5File(model_file, 'r')
    model = bob.learn.libsvm.Machine(f)
    del f

    return model


  ######################################################
  ################ Feature comparison ##################
  def score(self, model, probe):
    """Computes the score for the given model and the given probe.
    The score are Log-Likelihood.
    Therefore, the log of the likelihood ratio is obtained by computing the following difference."""
    assert isinstance(model, bob.learn.libsvm.Machine)
    self._check_feature(probe)

    # Generating supervector
    self.enroll_trainer = bob.learn.em.MAP_GMMTrainer(self.ubm, relevance_factor = self.relevance_factor, update_means = True, update_variances = False)
    map_feature = self.enroll_gmm(probe)
    mean_supervectors = map_feature.mean_supervector

    # Computing SVM score
    return model.predict_class_and_scores(mean_supervectors)[1]


  def score_for_multiple_probes(self, model, probes):
    raise NotImplementedError("Implement Me!")
