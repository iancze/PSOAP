try:
    from prior import prior
    print("Loaded user defined prior.")
except ImportError:
    print("Using default prior.")
    # Use default priors
    prior = None


prior(10)
