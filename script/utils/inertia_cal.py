# m = 2.4
# h = 0.09 
# r = 0.175

m = 2.4
h = 0.09 
r = 0.175

ixx = 1./12 * m * (3 * r**2 + h**2)
iyy = 1./12 * m * (3 * r**2 + h**2)
izz = 1./2 * m * r**2
print ixx, iyy, izz