import bob.bio.base

def add_jobs(args, submitter, local_job_adder):
  """Adds all (desired) jobs of the tool chain to the grid, or to the local list to be executed."""

  assert args.grid is not None

  # Here, we use the default bob.bio.base add_jobs function, but intercept it for adding the training
  SKIPS = ['preprocessing', 'extractor_training', 'extraction', 'projector_training', 'projection', 'enroller_training', 'enrollment', 'score_computation', 'concatenation', 'calibration']
  original_skips = {key : args.__dict__["skip_%s" % key] for key in SKIPS}

  # first, submit preprocessing and feature extraction; skip all others
  for key in SKIPS[3:]:
    setattr(args, "skip_%s" % key, True)

  job_ids = bob.bio.base.script.verify.add_jobs(args, submitter)

  for key in SKIPS[3:]:
    setattr(args, "skip_%s" % key, original_skips[key])

  # reset skips
  args.skip_preprocessing = original_skips['preprocessing']
  args.skip_extractor_training = original_skips['extractor_training']
  args.skip_extraction = original_skips['extraction']

  # if there are any external dependencies, we need to respect them
  deps = args.external_dependencies[:]
  # also, we depend on all previous steps
  for n in ['preprocessing', 'extractor-training', 'extraction']:
    if n in job_ids:
      deps.append(job_ids[n])

  # now, add our jobs
  job_ids, deps = local_job_adder(args, job_ids, deps, submitter)

  # alright, finish the remaining bits
  for key in SKIPS[:4]:
    setattr(args, "skip_%s" % key, True)

  args.external_dependencies = deps
  job_ids.update(bob.bio.base.script.verify.add_jobs(args, submitter))

  # alright, finish the remaining bits
  for key in SKIPS[:4]:
    setattr(args, "skip_%s" % key, original_skips[key])

  return job_ids