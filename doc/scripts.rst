
.. module:: psoap

.. _scripts:

Scripts
=======

The following scripts are made available on your command-line by the
``PSOAP`` package.

Initialization
--------------

.. code:: python

    !psoap-initialize --help


.. parsed-literal::

    usage: psoap-initialize [-h] [--check] [--model {SB1,SB2,ST3}]
    
    Initialize a new directory to do inference.
    
    optional arguments:
      -h, --help            show this help message and exit
      --check               To help folks check whether the package was installed
                            properly.
      --model {SB1,SB2,ST3}
                            Which type of model to use, SB1, SB2, ST1, or SB3.


.. code:: python

    !psoap-generate-chunks --help


.. parsed-literal::

    /usr/bin/sh: psoap-generate-chunks: command not found


.. code:: python

    !psoap-process-chunks --help


.. parsed-literal::

    /usr/bin/sh: psoap-process-chunks: command not found


.. code:: python

    !psoap-generate-masks --help


.. parsed-literal::

    /usr/bin/sh: psoap-generate-masks: command not found


.. code:: python

    !psoap-process-masks --help


.. parsed-literal::

    /usr/bin/sh: psoap-process-masks: command not found


Sampling
--------

The following scripts are used in sampling the posterior distribution.

.. code:: python

    !psoap-sample --help


.. parsed-literal::

    You need to copy a config.yaml file to this directory, and then edit the values to your particular case.
    Traceback (most recent call last):
      File "/home/ian/.build/anaconda/bin/psoap-sample", line 11, in <module>
        load_entry_point('psoap==0.0.1', 'console_scripts', 'psoap-sample')()
      File "/home/ian/.build/anaconda/lib/python3.6/site-packages/pkg_resources/__init__.py", line 570, in load_entry_point
        return get_distribution(dist).load_entry_point(group, name)
      File "/home/ian/.build/anaconda/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2751, in load_entry_point
        return ep.load()
      File "/home/ian/.build/anaconda/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2405, in load
        return self.resolve()
      File "/home/ian/.build/anaconda/lib/python3.6/site-packages/pkg_resources/__init__.py", line 2411, in resolve
        module = __import__(self.module_name, fromlist=['__name__'], level=0)
      File "/home/ian/.build/anaconda/lib/python3.6/site-packages/psoap-0.0.1-py3.6-linux-x86_64.egg/psoap/sample.py", line 33, in <module>
        f = open("config.yaml")
    FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'


.. code:: python

    !psoap-sample-parallel --help


.. parsed-literal::

    usage: psoap-sample-parallel [-h] [--debug] run_index
    
    Sample the distribution across multiple chunks.
    
    positional arguments:
      run_index   Which output subdirectory to save this particular run, in the
                  case you may be running multiple concurrently.
    
    optional arguments:
      -h, --help  show this help message and exit
      --debug     Print out debug commands to log.log

