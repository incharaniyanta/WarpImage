
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


def read_input_image(Imgpath):
    imgIn = cv2.imread(Imgpath)
    return imgIn


# In[3]:


def set_outimg_white(img):
    imgOut = 255 * np.ones(img.shape, dtype = img.dtype)
    return imgOut


# In[4]:


def in_triangle(dim):
    triIn = np.float32([dim])
    return triIn


# In[5]:


def out_triangle(dim):
    triOut = np.float32([dim])
    return triOut


# In[6]:


def calc_bounding_box(tri):
    r = cv2.boundingRect(tri)
    return r


# In[21]:


def crop():
    triCropped = []
    return triCropped


# In[22]:


def change_coordinates(tri1Cropped, tri1, r1, tri2Cropped, tri2, r2, img1):
    for i in xrange(0, 3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))
 
    # Crop input image
    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    return img1Cropped


# In[11]:


def calc_affine_trans(tri1Cropped, tri2Cropped):
    warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    return warpMat


# In[12]:


def apply_affine_trans(img1Cropped, warpMat, r2):
    img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return img2Cropped


# In[14]:


def get_mask(r2):
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    return mask


# In[16]:


def fillConvexPoly(tri2Cropped, mask):
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);
    img2Cropped = img2Cropped * mask
    return img2Cropped


# In[17]:


def copy_tri_to_output(img2, r2, img2Cropped):
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped
    
    


# In[19]:


def draw_triangle(a, b, c):
    color = (a, b, c)
    return color


# In[20]:


def draw_triangle_in_image(img, tri, color):
    cv2.polylines(img, tri.astype(int), True, color, 2, 16)


# In[ ]:


if __name__ == '__main__' :
      
   #Find bounding rectangle for each triangle
   r1 = calc_bounding_box(triIn)
   r2 = calc_bounding_box(triOut)
   
   # Offset points by left top corner of the respective rectangles
   tri1Cropped = crop()
   tri2Cropped = crop()
   
   img1Cropped = change_coordinates(tri1Cropped, triIn, r1, tri2Cropped, triOut, r2, imgIn)
   
   # Given a pair of triangles, find the affine transform.
   warpMat = calc_affine_trans(tri1Cropped, tri2Cropped)
   
   # Apply the Affine Transform just found to the src image
   img2Cropped = apply_affine_trans(img1Cropped, warpMat, r2)
   
   # Get mask by filling triangle
   mask = get_mask(r2)
   
   fillConvexPoly(tri2Cropped, mask)

   # Copy triangular region of the rectangular patch to the output image
   copy_tri_to_output(imgOut, r2, img2Cropped)
   
   # Draw triangle using this color
   color = draw_triangle(255, 150, 0)
   
   # Draw triangles in input and output images.
   
   draw_triangle_in_image(imgIn, triIn, color)
   draw_triangle_in_image(imgOut, triOut, color)
   
   cv2.imshow("Input", imgIn)
   cv2.imshow("Output", imgOut)
   
   
   cv2.waitKey(0) 
   # Read input image
   imgIn = read_input_image("C:/Desktop/robot.jpg")
   print(imgIn)
   
   # Output image is set to white
   imgOut = set_outimg_white(imgIn)
   
   # Input triangle
   triIn = in_triangle([[360,200], [60,250], [450,400]])
   
   # Output triangle
   triOut = out_triangle([[400,200], [160,270], [400,400]])
   
   # Warp all pixels inside input triangle to output triangle - begins
   
   #Find bounding rectangle for each triangle
   r1 = calc_bounding_box(triIn)
   r2 = calc_bounding_box(triOut)
   
   # Offset points by left top corner of the respective rectangles
   tri1Cropped = crop()
   tri2Cropped = crop()
   
   img1Cropped = change_coordinates(tri1Cropped, triIn, r1, tri2Cropped, triOut, r2, imgIn)
   
   # Given a pair of triangles, find the affine transform.
   warpMat = calc_affine_trans(tri1Cropped, tri2Cropped)
   
   # Apply the Affine Transform just found to the src image
   img2Cropped = apply_affine_trans(img1Cropped, warpMat, r2)
   
   # Get mask by filling triangle
   mask = get_mask(r2)
   
   fillConvexPoly(tri2Cropped, mask)

   # Copy triangular region of the rectangular patch to the output image
   copy_tri_to_output(imgOut, r2, img2Cropped)
   
   # Draw triangle using this color
   color = draw_triangle(255, 150, 0)
   
   # Draw triangles in input and output images.
   
   draw_triangle_in_image(imgIn, triIn, color)
   draw_triangle_in_image(imgOut, triOut, color)
   
   cv2.imshow("Input", imgIn)
   cv2.imshow("Output", imgOut)
   
   
   cv2.waitKey(0)

