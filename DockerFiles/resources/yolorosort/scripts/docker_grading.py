import cv2
import numpy as np
import os
import csv

#location of Pictures
pictures = "C:\\Users\\gerwi\\Dropbox\\uni\\BA_Erdbeeren\\Datasets\\exp\\crops\\"

#constants of HVS colours to find strawberries
mycolors = [
            [np.array([0,5,0]),np.array([61,229,150]), "1_1"],
            [np.array([14, 92, 119]), np.array([35, 200, 255]), "2_1"], # "green": color found from testenv.py using pics in StrawberyNotReady folder
            [np.array([0, 34, 67]), np.array([56, 255, 255]), "2_2"], # "green1": colour found from testenv.py using right0001963.jpg
            [np.array([32,10,42]),np.array([74,220,178]), "2_3"],
            [np.array([15,84,47]),np.array([104,224,223]), "3_1"],
            [np.array([3,17,162]),np.array([25,151,253]), "3_2"],
            [np.array([2,57,69]),np.array([19,211,255]), "3_3"],
            [np.array([0,50,55]),np.array([13,255,255]), "4_1"],     #"red": color found from testenv.py using colors.jpg
            [np.array([0,189,105]),np.array([20,255,255]), "4_2"],  #"red2": color found from testenv.py using right0004065.jpg
            [np.array([0,121,67]),np.array([7,255,255]), "4_3"],
            [np.array([0,198,103]),np.array([198,255,255]), "4_4"],
            [np.array([2,137,103]),np.array([34,255,239]), "5_1"]
            ]

#define constants:
pi = 3.1415
#pix = 1/50   #convert from pixel to cm -> 1 cm = 1/p Pixel is done in the getpix function
dens = 0.858  #Density of a  strawberry in g/cm³
folderpath = "/berry_pics"
'''preprocess function: 
Processes raw strawberry picture, finds all the contours and picks (??? and combines ???)
the best one by choosing the biggest then turnes it into useable format (black and white)
Input: Foto, list-colorsetting
Output: Image(red and black, (700,700)) and an initial grade TUPEL'''
def getrgbcontourimage(img, colors=mycolors):
    ''' Function processes Image into a black and white shape of the original,
    uncomment the linecomment with join, to get all the contours find with the parameters in colors'''
    dif = {}  #stores area as key and contour as value
    #join = np.zeros((img.shape[0], 1), np.uint8)
    hsvimage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurhsv = cv2.GaussianBlur(hsvimage, (5, 5), 0)  # optional, does not do that much, initial (5,5)
    for col in range(len(colors)):
########find strawberry in hsv plane
        mask = cv2.inRange(blurhsv, colors[col][0], colors[col][1])
        edgeimage = cv2.Canny(mask, 50, 100, 20)
        kernel = np.ones((2, 2))  # initial 5,5 smaler kernel works better with smaller pictures
        dialimage = cv2.dilate(edgeimage, kernel, iterations=2)  # initial 2
        first = cv2.erode(dialimage, kernel, iterations=1)  # initial 1
        #join = np.hstack((join, mask))
        #cv2.imshow("finder", join); cv2.waitKey(0)

########find maximum contour of this colour setting:
        contours, hierachy = cv2.findContours(first, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        areas = {}
        try:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                areas.update({area: cnt})
            sec = areas[max(areas.keys())]
        except ValueError as msg:
            sec = 1.0

########consider empty contours bc bad color finding setting:
        if type(sec) != np.ndarray:
            sec2 = np.array([(1, 1), (0, 0), (0, 1), (1, 0)], dtype=np.int32)
            val2 = float(1)
            #sec2 = np.array([(col,col), (0,0), (0,col), (col,0)], dtype=np.int32)
            #val2 = float(col)
            dif.update({val2: [sec2,colors[col][2]]})
            #print("empty contour replaced", type(sec), colors[col][2])
        else:
            val = cv2.contourArea(sec)
            dif.update({val:[sec,colors[col][2]]})
########save the maximum contour, move a little from border by adding 100 (estimate):
    maxcont = max(dif.keys())
    cont = dif[maxcont][0]
    grade = dif[maxcont][1]
    #print(type(grade))
    #cv2.imwrite("Masksbycolorconstants.jpg", join) #save masks

########Draw found contour as filled space onto black image that is big enough for pic to turn (700,700):
    blank = np.zeros(img.shape, np.uint8) #black image

    #imagergb = cv2.drawContours(img.copy(), cont, -1, (255, 255, 255), 2)
    imagergb = cv2.fillPoly(blank, pts=[cont], color=(0,0,255)) #white colour
    finalimage = cv2.copyMakeBorder(imagergb,100,100,100,100,cv2.BORDER_CONSTANT,value=0)
    #cv2.imshow("finder", imagergb); cv2.waitKey(0)

########Draw fill the contour
    return finalimage, grade

'''Get contour function: 
takes the prprocessed RGB image and returns a tupel with 
0 beeing the image itself and 1 beeing the shape of the image as contour
Input: Image(red&black)
Output: tuple-image(red&black),Contour(corresponding)'''
def getcontour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grays out the image
    # Find contours
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    return (image,c)

'''Convex Hull function: 
takes the preprocessed image and changes the shape of the strawberry into a convex form and 
returns an image in black and red of that new form
Input: image(red&black)
Output: Image(red&black,"convex")'''
def convexHull(set):
    hull = cv2.convexHull(set[1])
    newimage = cv2.fillPoly(set[0].copy(), pts=[hull], color=(0,0,255))
    return newimage

'''Center of Mass function:
Function that takes an image that is only red (255,0,0) and black (0,0,0) and returns the x and y coordinate
(y=608, x=504, depth=3) of the center of mass based, so that the number of pixels above and below in both x and y are the same
Input: Image(Red&black)
Output: tuple-X&Y from center of mass'''
def getcenterofmass(image):
    #Convert to grayscale
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Threshold via Otsu + bias adjustment:
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply an erosion + dilation to get rid of small noise:

    # Set kernel (structuring element) size:
    kernelSize = 3

    # Set operation iterations:
    opIterations = 3

    # Get the structuring element:
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

    # Perform closing:
    openingImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    openingImage = cv2.bitwise_not(openingImage)

    # Calculate the moments
    imageMoments = cv2.moments(openingImage)

    # Compute centroid, if no centroid given, take center of image
    #cv2.imshow("centerpic", image); cv2.waitKey(0)
    try:
        cx = int(imageMoments['m10'] / imageMoments['m00'])
        cy = int(imageMoments['m01'] / imageMoments['m00'])
    except ZeroDivisionError as msg:
        cx = image.shape[1]//2
        cy = image.shape[0]//2
        print(msg)

    # return points
    return (cx,cy)

'''Find lowest Extrempoint Function:
Function takes a set of image and corresponding contour and returns lowest point of the contour
Input: tuple-image(red&black), Contour(corresponding)
Output: tuple-X&Y from lowest point red'''
def findlowestpoint(set):
# Obtain outer coordinates
    c = set[1]
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    return bottom

'''Rotate to lowest Function:
Function takes an image and the corresponding center of mass and Lowest point and rotates it until both of the points are
aligned vertically.
Input: (Image, tuple-centerofmass, tuple-lowestpoint)
Output: (Image, int(angle))'''
def rotate_image(image, centerofmass, lowestpoint):
####Calculate Vectors from Center of Mass to top and from Center of Mass to lowest point
    vector1 = [0, 0 - list(centerofmass)[1]]
    vector2 = [list(lowestpoint)[0]-list(centerofmass)[0], list(lowestpoint)[1]-list(centerofmass)[1]]

####Unify vectors to use dotproduct
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2) #calculates the shortest angle -> angle will always be <0

####transfer Rad into Deg and get the right angle, angle < 0 for left and angle > 0 for right turns
    angle = (1 - np.arccos(dot_product) / pi) * 180
    if vector2[0] >= 0: #lowest point on RIGHT side of center of mass
        angle *= -1
    else:#lowest point on LEFT side of center of mass
        pass

####get rotation Matrix and perform turn
    rot_mat = cv2.getRotationMatrix2D(centerofmass, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

####Ouputs an image that is turned
    return result, angle

def rotimgtolowest(image):
    '''Function takes a preprocessed image (red and black, convexHull) and turns back and forth until the "pointy" end is
       the lowest point. Used Functions: rotate_image, findlowestpoint, getcenterofmass
        Input: Image(red&black, "convex")
        Output: Image(red&black, "convex-turned"'''
    stopp = 1
    rotimg = image.copy()
    while stopp <= 3:
        # get center of mass and lowest point:
        centerofmass = getcenterofmass(rotimg)
        step1 = getcontour(rotimg)
        lowestpoint = findlowestpoint(step1)

        # draw both points:
        #step2 = cv2.circle(rotimg, centerofmass, 2, (0, 255, 255), -1)
        #rotimg = cv2.circle(step2, lowestpoint, 2, (255,255,0), -1)

        # turn picture so that center of mass and lowest point align vertically
        step4 = rotate_image(rotimg, centerofmass, lowestpoint)
        stopp += 1
        if int(step4[1]) == 0: break
        else: rotimg = step4[0]

    # return the rotated image
    return step4[0]

'''Volume Function: function calls function
Function takes an image, calculates the center of mass, the distance between it and the center of the two parts
left and right of the main center of mass and the Area of red particels in each half. Then using the first rule of 
Guldini to create a rotation Volume by rotating both sides for pi. Resulting in an float value for the Volume in 
cubig pixels
Input: Image(red&black, "vertically aligned")
Output: Float-Volume in cubic pixels
'''
def calculatemass(image):

    pix = getpix(image.shape, 26, 1.847963623046875,1.848005615234375,1.9210474853515625,1.0994735107421875)

    center = getcenterofmass(image)[0]

    #cut image in half where center of mass is
    imageright = image[:,center:]
    imageleft = image[:,:center]

    # calculate the distance between center of mass of right and left side relatif to main center of mass
    centerright = getcenterofmass(imageright)[0]
    centerleft = getcenterofmass(imageleft)[0]
    distright = centerright * pix
    distleft = (center - centerleft) * pix

    #calculate area of both sides
    arearight = countpixels(imageright) * (pix ** 2)
    arealeft = countpixels(imageleft) * (pix ** 2)

    # calculate first volume of right and left, if one is very small (under 10%), calculate all with big side
    if arealeft/(arealeft + arearight) <= 0.1:
        volumeleft = pi * distright * arearight
    else: volumeleft = pi * distleft * arealeft

    if arearight/(arealeft + arearight) <= 0.1:
        volumeright = pi * distright * arealeft
    else: volumeright = pi * distright * arearight


    mass = (volumeright + volumeleft)*dens

    return mass

def getpix(shape, depth, fy, fx, cx, cy,):
    #fx, fy and cx, cy are found by calibrating the lens, depth comes from previous steps

    #get reflexion length in pixel from original picture shape
    ux = (shape[0] - 200)//2
    uy = (shape[1] - 200)//2

    # calculate real length of both horizontal and vertical length
    realx = (ux-cx) * depth / fx
    realy = (uy-cy) * depth / fy

    # calculate verticaL and horizontal pixel value
    pixv = ux/realx
    pixh = uy/realy

    # calculate pix value
    pix = (pixv + pixh)/2

    return pix

def countpixels(image):
    #calculate area of both sides
        #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #calculate area
    area = cv2.countNonZero(gray)

    # optional comparisson with contourArea
    # area2 = cv2.contourArea(getcontour(image)[1])
    # print("Count zeros", area, "contour area", area2, area/area2, sep="\t")

    return area

'''Process all steps function:
Fnction that takes one image and processes it: first finds a shape, then makes the shape convex, then turns it and 
finds is volume and mass
Input: image(from best pic, after getrgbcontour), list of lists-color constants
Output: tuple of tuples-(original image, first step, scond step, thrid step),(mass)'''
def processallsteps(image,third):
    # convert image into rgb in red and black
    first = image[0] #original red shape
    inigrade = int(image[1][0])

    # change contour of rgb image into convexHull image
    # taken from best pic function third == convexHull

    # test rotatetolowest function:
    forth = rotimgtolowest(third)

    # calculate Volume and print it out
    #print(count, calculatemass(forth), "Mass in g", sep="\t")

    # display all images
    #image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=0)
    mass = calculatemass(forth)

    return ((first,third,forth),mass,inigrade)

'''find best picture function:
Function detects the best picture by calculating the difference between the original shape and the convex hull, the one
with the least is the best picture
Input: image, red and black with original shape from getrgbcontour, id formatted to a 5 digit number, color constants
Output: newly graded and saved image as ID_MASS_GRADE_DIFFERENCE.jpg
'''
def bestpic(image, id):
    # check if existing file
    # load all the files in the right folder
    files = os.listdir(folderpath)
    default = "#" * 19 + "10000" + ".txt"
    files.append(default)
    # look at difference of existing picture
    for f in files:
        if f[:5] == id or f[:5] == "#"*5:
            dif1 = int(f[19:-4])
            org = folderpath + "/" + f

        #use all the processing functions and save accordingly
        # calculate area of raw found shape
            arearaw = countpixels(image[0])

        # calculate area of the convex hull picture
            sec = getcontour(image[0].copy())
            third = convexHull(sec)
            areaconvex = countpixels(third)

        # calculate difference
            dif = areaconvex - arearaw

        # save if lowest dif
            if dif < dif1:
                if dif1 == 10000:
                    best = processallsteps(image,third)
                    return (best,dif)
                else:
                    os.remove(org)
                    best = processallsteps(image,third)
                    return (best,dif)
            else:
                return ("worse quality",dif)

'''Grade strawberies Function:
Function takes the processed data of the best picture and uses the convex picture to create a mask and lay it over the 
original image. It then uses predefined constants, 5 in total, to distinguish the ripeness
1...white color in the picture -> still a flower
2...dominant green in the picture -> still a green Strawberry
3...dominant light red in the picture -> almost ready Strawberry
4...dominant red in the picture -> perfect Strawberry
5...dominant darker red in the picture -> overly ripe Strawberry
Input: tuple of tuples: ((img,image,image,image),float)
Output: integer'''
def grade(imagetuple):
    # colorconstants: list of lists containing: 0_colorconstantlower, 1_colorconstantupper, 2_number of state, 3_description of state
    colorconstants = [
        [np.array([0, 0, 200]), np.array([179, 45, 255]), 1, "still a Flower"],
        [np.array([25, 46, 150]), np.array([35, 106, 255]), 2, "still green"],
        [np.array([0, 79, 180]), np.array([14, 172, 255]), 3, "almost ready"],
        [np.array([0, 184, 100]), np.array([10, 255, 255]), 4, "perfect Strawberry"],
        [np.array([0, 0, 39]), np.array([10, 255, 150]), 5, "old Strawberry"]
    ]
    dominant = {0:0,1:0,2:0,3:0,4:0,5:0}

    # get the contour and original image, get rid of the black boundary
    cut = 100
    scale = 4
    conveximage = imagetuple[0][2][cut:-cut,cut:-cut]
    conveximage = cv2.resize(conveximage, (int(conveximage.shape[1]*scale),int(conveximage.shape[0]*scale)))
    original = imagetuple[0][0]#[cut:-cut,cut:-cut]
    original = cv2.resize(original, (int(original.shape[1] * scale), int(original.shape[0] * scale)))

    # find contour of the convex form
    contour = getcontour(conveximage)[1]

    # create mask of contour and only show part of original picture in that mask
    stencil = np.zeros(original.shape).astype(original.dtype)
    stencil = cv2.fillPoly(stencil, pts=[contour], color=(255, 255, 255))
    result = cv2.bitwise_and(original, stencil)
    #cv2.imshow("scoped", result); cv2.waitKey(1)

    # preprocess picture to check for colour constants
    hsvimage = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsvimage = cv2.GaussianBlur(hsvimage, (5, 5), 0)  # optional mit step 40 gut

    # create step loop vertical
    step = 50
    strip = 5 #shades on borderless images can not be found -> prevents that

    for part in range(step,hsvimage.shape[0],step):
        #stripe ... stripe of the picture to be checked for colour either left to right or top to bottom
        stripe = hsvimage[part-step:part,:] #from top to bottom - [:,part-step:part] from left to right
        stripe = cv2.copyMakeBorder(stripe,strip,strip,strip,strip,cv2.BORDER_CONSTANT,value=(255,255,255))

    # create loop through colour constants for each stripe
        for color in colorconstants:
            mask = cv2.inRange(stripe, color[0], color[1])
            #cv2.imshow("stripe", mask); cv2.waitKey(0); cv2.destroyAllWindows()
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            area1 = 0
            if cnts == ():
                area1 += 0
            else:
                for cont in cnts:
                    area1 += cv2.contourArea(cont)


            up = {color[2]:area1}
            dominant.update(up)
    #print(dominant.values())
    grd = [k for k, v in dominant.items() if v == max(dominant.values())]
    #print(grd)
    if len(grd) == 1:
        grade = grd[0]
    else: grade = int(imagetuple[1])

    #retrun just the grading number
    return grade


