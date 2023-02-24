import numpy as np

""" q_value_average.py is used to calculate the average value of the order Q and the constant values
    for quadratic and linear convergence """

q_cg = [  1.0003018,    1.00029481,   1.00021643,   1.00024257,   1.00013762, 1.00022855,   1.00018569,   1.00018698,   1.00019842,   1.00017657 ]
q_dl = [  1.00004653,   1.00004916,   1.00004578,   1.0000454,    1.00004576,
    1.00004683,   1.00004519,   1.00004656,   1.0000458,    1.0000476 ]
q_cp =  [  1.02942627,   1.02648243,   1.02270741,   1.02365466,   1.02056511,
    1.01928517,   1.01710065,   1.01699026,   1.01668591,   1.01586812]


c_quad_cg = [  0.99690859,   0.99684808,   0.99632733,   0.99637233,   0.99606172,
    0.99622362,   0.99597666,   0.99597037,   0.99600852,   0.99590256]
c_quad_dl = [  0.9272451,    0.92646548,   0.91907262,   0.91775351,   0.91688697,
    0.91590626,   0.91266207,   0.91662592,   0.91429317,   0.91565504]
c_quad_cp = [  0.9997069,    0.99970786,   0.99965617,   0.99967932,   0.99968598,
    0.99969252,   0.99966546,   0.99967855,   0.99967837,   0.99967455]

c_lin_cg = [  0.99690859,   0.99684808,   0.99632733,   0.99637233,   0.99606172,
    0.99622362,   0.99597666,   0.99597037,   0.99600852,   0.99590256]

c_lin_dl = [  0.9272451,    0.92646548,   0.91907262,   0.91775351,   0.91688697,
    0.91590626,   0.91266207,   0.91662592,   0.91429317,   0.91565504]
c_lin_cp = [  0.9997069,    0.99970786,   0.99965617,   0.99967932,   0.99968598,
    0.99969252,   0.99966546,   0.99967855,   0.99967837,   0.99967455]

mean_q_cg = np.mean(q_cg)
mean_q_cp = np.mean(q_cp)
mean_q_dl = np.mean(q_dl)

mean_quad_c_cg = np.mean(c_quad_cg)
mean_quad_c_dl = np.mean(c_quad_dl)
mean_quad_c_cp = np.mean(c_quad_cp)

mean_lin_c_cg = np.mean(c_lin_cg)
mean_lin_c_dl = np.mean(c_lin_dl)
mean_lin_c_cp = np.mean(c_lin_cp)

print(f"The mean value of Q for Steihaug method is : {mean_q_cg}")
print(f"The mean value of Q for Dog-leg method is : {mean_q_dl}")
print(f"The mean value of Q for Cauchy method is : {mean_q_cp}")

print(f"The Quad C for Steihaug method is : {mean_quad_c_cg}")
print(f"The Quad C for Dog-leg method is : {mean_quad_c_dl}")
print(f"The Quad C for Cauchy method is : {mean_quad_c_cp}")

print(f"The lin C for Steihaug method is : {mean_lin_c_cg}")
print(f"The lin C for Dog-leg method is : {mean_lin_c_dl}")
print(f"The lin C for Cauchy method is : {mean_lin_c_cp}")
