cd code_doc
sphinx-apidoc -o . ../src
sphinx-apidoc -o ./extras ../src/extras
sphinx-apidoc -o ./data ../src/data
sphinx-apidoc -o ./data_statistics ../src/data_statistics
sphinx-apidoc -o ./models ../src/models
sphinx-apidoc -o ./optimizers ../src/optimizers
sphinx-apidoc -o ./physics_modules ../src/physics_modules
sphinx-apidoc -o ./reproduce_modules ../src/reproduce_modules
make html