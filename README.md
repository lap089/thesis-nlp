# thesis-nlp

The project applies nlp techniques for thesis application

# Project structure
- training: contains binary files for training (more than 4 mil wikipedia articles)
- model: contains binary model file after training process.
- train.py: train
- convert.py: convert document to vector



# Integration
- Import convert.py file into your project.
- Init NewsConverter class for loading Doc2Vec model
- Use convert_doc_to_vector function to convert a string to vector
- Note: remember to copy model folder since it contains binary file for loading data.




