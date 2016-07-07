# Summer-Internship
This is a collection of codes for the objective of predicting keywords in a given text. The approach uses Recurrent Neural Networks(RNNs) to learn to identify the keywords based on the context. Torch and its libraries have been used to code the neural nets. Following is a brief description of all the files:

* [forward.lua](blob/master/forward.lua) - implements the basic architecture, takes in the filenames and stores a model generated after each epoch.
2. backward.lua - same as forward.lua except it reverses the input and gives it to the network.
3. training.lua - converts the data from csv format into Torch Table and dumps it for future use.
4. vector_forward.lua - almost same as forward.lua except that it includes a lookuptable layer (embedding) in the model, which is initialised with glove embedding and allowed to update.
5. murphy_vector_forward.lua - runs vector_forward.lua on "Murphy" book.
6. vector_backward.lua - analogous to vector_forward.lua with reversed input.
7. murphy_vector_backward.lua - runs vector_backward.lua on "Murphy" book.
8. feature_forward.lua - similar to vector_forward.lua but includes three LookUp tables (word embedding, POS tag and Chunk Tag), one for every feature.
9. validation.lua - runs the validation/testing data on every model generated, picks up the best one, and then reports the performance parameters on the model, for various thresholds.
10. cross_validation.lua - runs the model generated by one book on test data of the other.
11. fwd_validation.lua, bwd_validation.lua - just a particular instance of validation.lua, where the input is fed in either forward or backward fashion respectively.
12. vector_fwd_validation.lua, vector_cross_validation.lua, vector_cross_validation.lua - vector update analogue of fwd_validation, bwd_validation.lua and cross_validation.lua respectively.
