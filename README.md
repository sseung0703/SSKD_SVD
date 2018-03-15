Self-supervised Knowledge Distillation using Singular Value Decomposition

Abstract. To solve deep neural network (DNN)'s huge training dataset and its high computation issue, so-called teacher-student (T-S) DNN which transfers the knowledge of T-DNN to S-DNN has been proposed[1].
However, the existing T-S-DNN methods(s) has limited range of use, and the knowledge of T-DNN is not suffciently transferred to S-DNN. 
In order to improve the quality of the transferred knowledge from T-DNN, we propose a new knowledge distillation method using singular value de-composition (SVD).
In addition, we define a knowledge transfer as a self-supervised task and suggest a way to continuously receive information from T-DNN.
Simulation results show that a S-DNN with a computational cost of 1/5 of the corresponding T-DNN can be up to 1.1% better than the T-DNN in terms of classiffcation accuracy.
Also assuming the same computational cost, our S-DNN outperforms the S-DNN driven by the state-of-the-art distillation method with a performance advantage of 1.79%.
