#!/usr/bin/env python
# coding: utf-8

# Data Science Workshop-1 (CSE 2195)
# ASSIGNMENT-5: NUMPY
# ABHA MAHATO
# 2241013032
# 1. Find the index of the 5th repetition of number 1 in 
# x= np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2]).
# 

# In[24]:


import numpy as np

x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
a=np.where(x==1)[0]
if len(a)>=5:
    a5=a[4]
    print("index at fifth position: ",a5)


# 2. Compute the Euclidean distance between two arrays a = np.array([1,2,3]) b = np.array([4,5,6])

# In[23]:


a=np.array([1,2,3])
b = np.array([4,5,6])
distance=np.sqrt(np.sum((b-a)**2))
print("distance withoput using in built function:",distance)
distance = np.linalg.norm(a - b)
print("distance  using in built function:",distance)


# 3. (a) Replace all odd numbers in the array with -1 without changing the array.

# In[21]:


a=np.array([1,2,3,4,5,6,7,8,9])
mod_array=np.where(a%2!=0,-1,a.copy())
print("Original array: ",a)
print('Modified array: ',mod_array)


# (b)From the array a, replace all values greater than 30 to 30 and less than 10 to 10.

# In[26]:


a = np.array([5, 15, 25, 35, 10, 20, 30, 40])
a[a>30]=30
a[a<10]=10
print(a)


# 4. Create a 4*4 matrix with entries from uniform distribution data in the interval 10 to 20. Normalize
# the matrix so that the minimum has a value of 0 and the maximum has a value of 1.

# In[54]:


s=np.random.uniform(10,20,size=(4,4))
print("Array: ",s)
ma=(s-np.min(s)/(np.max(s)-np.min(s)))
ma


# 5. Find the mean, median, and standard deviation of a 1-d array

# In[55]:


x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1])
mean=np.mean(x)
median=np.median(x)
std=np.std(x)
print("mean ",mean)
print("median ",median)
print("standard deviation ",std)


# 6. Get all items between 5 and 10 from a. Input: a= [2, 6, 1, 9, 10, 3, 27]. 

# In[66]:


r= [2, 6, 1, 9, 10, 3, 27]
a=np.array(a)
s=(a>=5) &(a<=10)
r1=a[s]
print(r1)


# 7. Convert a 1D array to a 2D array with 2 rows

# In[7]:


import numpy as np
a=np.random.randn(10)
print("1D ",a)
b=a.reshape(2,-1)
print("2D: ",b)


# 8. Create a 2D array and swap the first two rows and two columns

# In[15]:


x=np.arange(1,10).reshape((3,3))
swap_row=x[[1,0,2],:]
swap_col=swap_row[:,[1,0,2]]
print("array: \n",x)
print("swapped array:\n",swap_col)


# 
# 9. Find the most frequent values in an array of positive integers. The original array is [6 9 5 1 7 5 1 0 1
# 5 5 0 8 9 0 7 0 7 6 5 1 1 9 5 3 8 7 9 6 3 4 5 9 7 2 7 0 2 2 6].

# In[23]:


a=np.array([6,9,5 ,1 ,7, 5 ,1, 0 ,1,5 ,5 ,0, 8, 9, 0 ,7, 0 ,7, 6, 5, 1, 1, 9, 5 ,3, 8 ,7, 9, 6, 3, 4, 5, 9, 7 ,2 ,7, 0 ,2 ,2, 6])
un,counts=np.unique(a,return_counts=True)
max_count = np.argmax(counts)
mostn=un[counts==counts[max_count]]
print("most frequent no: ",mostn)


# 10. Create a symmetric matrix of order 4*4, whose items are taken from a standard normal distribution.

# In[24]:


m=np.random.randn(4,4)
m=(m+m.T)/2
print("symmetric_matrix: ",m)


# 11. In a cricket match, a batsman scores any one of {1, 2, 3, 4, 6, 0}. When he scores 0, it will be considered as ”OUT”. If he will face 50 balls maximum, then find the score that he will make before getting
# ”OUT”. Draw the plot of the run scored by the batsman

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
balls_faced = 0
total_score = 0
runt= []
while balls_faced < 50:
   run = np.random.choice([1, 2, 3, 4, 6, 0])  
   runt.append(run)
   
   if run == 0:
       break  
   else:
       total_score += run
       balls_faced += 1
plt.plot(np.cumsum(runt), marker='o')
plt.title('Run Sequence of the Batsman')
plt.xlabel('Ball Number')
plt.ylabel('Runs Scored')
plt.show()
print(runt)
print(np.cumsum(xlabel,ylabel))
print("Total runs scored before getting out:", total_score)
print("Balls faced:", balls_faced)
   


# 12. Compute the cross product of two 3*3 matrices

# In[48]:


x=np.arange(1,10).reshape((3,3))
y=np.arange(11,20).reshape((3,3))


crossR=np.cross(x[:,None,:],y,axisb=0,axisc=0)

print(x)
print(y) 
print(crossR)


# 13. Write a NumPy program to sort the student id with increasing height of the students from the given
# student id and height. Print the integer indices that describe the sort order by multiple columns and
# the sorted data.

# In[50]:


student_data = np.array([
    [2241011101, 175],
    [2241012102, 160],
    [2241017103, 180],
    [2241013032, 165],
    [2241011211, 172]
])
sorted_indices = np.lexsort((student_data[:, 1], student_data[:, 0]))
sorted_data = student_data[sorted_indices]
print(sorted_data)


# 14. Return array of odd rows and even columns from below numpy array :
# x= np.array([[3, 6, 9, 12], [15, 18, 21, 24], [27, 30, 33, 36], [39, 42, 45, 48], [51, 54, 57, 60]]

# In[51]:


x = np.array([[3, 6, 9, 12],
              [15, 18, 21, 24],
              [27, 30, 33, 36],
              [39, 42, 45, 48],
              [51, 54, 57, 60]])
result_array = x[::2, 1::2]
print(result_array)


# 15. Calculate the sum of all rows and columns of a 2-D NumPy array
# 

# In[53]:


a=np.arange(1,10).reshape((3,3))
sum_rows = np.sum(a, axis=1)
sum_columns = np.sum(a, axis=0)
print("Sum of all rows:")
print(sum_rows)

print("\nSum of all columns:")
print(sum_columns)


# 16. Write a code to multiply a 5x3 matrix by a 3x2 matrix and create a real matrix product.
# 

# In[58]:


x1=np.arange(1,16).reshape((5,3))
x2=np.arange(7,13).reshape((3,2))
p=np.dot(x1,x2)
print(p)


# 17. Write a code to find the roots of the polynomials x2 − 4x + 7 = 0.

# In[59]:


# Coefficients of the polynomial
coefficients = [1, -4, 7]

# Find the roots of the polynomial
roots = np.roots(coefficients)

print("Roots of the polynomial:")
print(roots)


# 18. Write a code to create a 5x5 array with random values and normalize it row-wise.

# In[61]:


a = np.random.random((5, 5))
na = (a.T / np.sum(a, axis=1)).T
print(na)


# 19. Write a NumPy program to create a (9*9*2 )array with random values and extract any array of shape
# (5,5,2) from the said array.

# In[46]:


s=np.random.standard_normal(size=(9,9,2))
print(s)
a= s[:5,:5,:2]
a


# 20. Write a code to create a 4*4 array with random values and sort each column.

# In[41]:


#20
s=np.random.standard_normal(size=(4,4))
print("array :",s)
sa=np.sort(s,axis=0)
print("sorted array :",sa)

