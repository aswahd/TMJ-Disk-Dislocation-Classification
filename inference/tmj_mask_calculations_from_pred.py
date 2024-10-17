#Code to read masks segmented from TMJ MRI ------ CENTER POINT DISC: DONE, LINE INTERSECTION: DONE, NORMAL LINE FROM CENTER TO LINE: IN PROGRESS
#Data Folder 
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance
import re
import cv2
import math
import openpyxl
from operator import itemgetter
import statistics
from progress.bar import Bar


width, height = 256,256
pred_folder = 'output_peds_model'
#Distance between point and a line
#d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
#perpendicualr point of intersection

def getShortestDistance(s1,s2):
    sd = distance.cdist(s1,s2)
    #sd = distance.cdist(s1,s2).min(axis=1)
    x,y = np.where(sd == np.min(sd))
    return(s1[x[0]],s2[y[0]])

 
def has_common_data(list1, list2):
    result = 'N'
    common_list = []
    common = 0

    # Removing duplicate points so there's no double counting
    list1, list2 = set(list1), set(list2)
    list1, list2 = list(list1), list(list2)
    # traverse in the 1st list
    for x in list1:
 
        # traverse in the 2nd list
        for y in list2:
   
            # if common element
            if x == y:
                common_list.append(y)
                common += 1
    if common >= 1: # Might change the number
       result = 'Y'          
    return result, common_list

def b_line(x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line

def get_mean_point(list_of_tuples):
    '''
    list_of_tuples contain items in the following manner: (x coordinate, y coordinate)
    Sift through list_of_tuples and calculate the mean x coordinate and the mean y coordinate, both rounded to the nearest integer
    Return a tuple with two points as follows: (x_mean, y_mean)
    ''' 

    if len(list_of_tuples) > 0:
        x_mean = round(sum(item[0] for item in list_of_tuples) / len(list_of_tuples))
        y_mean = round(sum(item[1] for item in list_of_tuples) / len(list_of_tuples))
        return (x_mean, y_mean)
    return "Empty List"

def get_extrema_point(list_of_tuples, extrema_wanted):
    '''
    Find the maxima or minima (based on the y value) and return the coordinates of the point as a tuple (x, y_extrema
    Note: Lower numerical value means higher up on the canvas, Higher numerical value means lower on the canvas.
    '''
    
    if extrema_wanted == 'max':
        return min(list_of_tuples, key=itemgetter(1)) # Highest point has the lowest y value hehe
    elif extrema_wanted == 'min':
        return max(list_of_tuples, key=itemgetter(1))

def get_extrema_point_x(list_of_tuples, extrema_wanted):
    '''
    Find the maxima or minima (based on the y value) and return the coordinates of the point as a tuple (x, y_extrema)
    Note: Numerical value is consistent with left and right.
    '''

    if extrema_wanted == 'min':
        return min(list_of_tuples, key=itemgetter(0))
    elif extrema_wanted == 'max':
        return max(list_of_tuples, key=itemgetter(0))
        
def get_distances_between_two_points(point1, point2):

    x_distance = abs(point1[0] - point2[0])
    y_distance = abs(point1[1] - point2[1])
    e_distance = math.sqrt(math.pow(x_distance, 2) + math.pow(y_distance,2)) # c^2 = a^2 + b^2


    return (x_distance, y_distance, e_distance)

def make_emcon_line(c_eminence, c_condyle, left_low_disc_point):

    # Get rid of dupes because they mess up calculations
    c_eminence, c_condyle = set(c_eminence), set(c_condyle)
    c_eminence, c_condyle = list(c_eminence), list(c_condyle)
    
    updated_c_eminence = []

    # Keep all eminence points that are to the right of the leftest most point of the disk to avoid early peaks
    for point in c_eminence:
        if point[0] >= left_low_disc_point[0]:
            updated_c_eminence.append(point)
            
    highest_eminence_point = get_extrema_point(updated_c_eminence, 'max')
    
    cleaned_c_eminence = []
    for point in updated_c_eminence:
        if point[0] <= highest_eminence_point[0]:
            cleaned_c_eminence.append(point)

    emcon1, emcon2 = getShortestDistance(c_condyle, cleaned_c_eminence) 
                        
    # Find all the points in between that is required to construct a line connecting the two closet points
    emcon_line_points = b_line(emcon1[0],emcon1[1],emcon2[0], emcon2[1])
    
    # Find x,y,e distances of the eminence-condyle line
    emcon_x_len,  emcon_y_len,  emcon_e_len = get_distances_between_two_points(emcon1,emcon2)

    return (emcon_line_points, [emcon_x_len,  emcon_y_len,  emcon_e_len])
    
def make_disc_line(c_disc):
    '''
    Anything related to making the disc line and calculations associated should be placed here
    By using this function, names can be shortened and it's implied that each variable is related to disc in some way!
    '''

    # Get rid of duplicates in c_disc and then turn back to list
    c_disc = set(c_disc)
    c_disc = list(c_disc)
    
    # Find the max and min y values
    # y_max = get_extrema_point(c_disc, 'max')[1] # Numerically lower
    # y_min = get_extrema_point(c_disc, 'min')[1] # Numerically higher

    # Alternate points
    top_right_x = get_extrema_point_x(c_disc, "max")[0] # Encapsulation, max or min are based on cartesian plane, top == right == max and vice_versa
    top_right_y = get_extrema_point(c_disc, "max")[1]
    
    bot_left_x = get_extrema_point_x(c_disc, "min")[0]
    bot_left_y = get_extrema_point(c_disc, "min")[1]

    top_right_point = (top_right_x, top_right_y)
    bot_left_point = (bot_left_x, bot_left_y)

    x = [p[0] for p in c_disc]
    y = [p[1] for p in c_disc]

    #find line of best fit
    a, b = np.polyfit(x, y, 1)
    
    best_fit_points = []
    for x_val in np.array(x):
        y_val = round(a * x_val + b)
        best_fit_points.append((x_val, y_val))

    top_right_best = get_extrema_point(best_fit_points, 'max')
    bot_left_best = get_extrema_point(best_fit_points, 'min')

    # # Find the distance between top_right and bot_left points
    disc_length = get_distances_between_two_points(top_right_best, bot_left_best)[2]
    
    # # Construct a list of points required for vertical disc line
    disc_line_points = b_line( bot_left_point[0],  bot_left_point[1],  top_right_point[0],  top_right_point[1])

    return (best_fit_points, disc_length, top_right_best, bot_left_best)

def find_yellow_ratio(c_disc, emcon_line_points, pn, sn):
    '''
    Variable name Legend: 
        lz = less than zero
        gz = greater than zero
        d = determinant
    '''
    # Get rid of dupes
    c_disc = set(c_disc)
    c_disc = list(c_disc)
    
    # # Do The Ratio, subtract the points on the line from the calculation
    dez = []
    dlz = []
    dgz = []

    ep1  = get_extrema_point(emcon_line_points, 'max')
    ep2 = get_extrema_point(emcon_line_points, 'min')
    # d=(x−x1)(y2−y1)−(y−y1)(x2−x1)
    for point in c_disc: # (x1,y1) = left_low, (x2,y2) = right_high, play around to see which sign convention works for my task
        
        
        # Got the idea from here https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
        d = (point[0] - ep1[0])  * (ep2[1] - ep1[1]) - (point[1] - ep1[1]) * (ep2[0] - ep1[0])

        if d == 0: # Skip points on the line but tally up their apperance with middle_count
            dez.append(point)
            continue   
            
        # For this dataset, it seems like d > 0 is on the left of the line
        if d > 0:
            dgz.append(point)
        elif d < 0:
            dlz.append(point)

    xlz = [p[0] for p in dlz]
    ylz = [-abs(p[1]) for p in dlz]

    xgz = [p[0] for p in dgz]
    ygz = [-abs(p[1]) for p in dgz]

    lx = [p[0] for p in emcon_line_points]
    ly = [-abs(p[1]) for p in emcon_line_points]

    fig, ax = plt.subplots(figsize =(2, 2))
    
    ax.scatter(xgz, ygz, label='d > 0')
    ax.scatter(lx,ly, label='line')
    ax.scatter(xlz, ylz, label='d < 0')
   
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

    fig1 = plt.gcf()
    # plt.show()
    plt.draw()
    fig1.savefig(f"./preds/mask_ratio_plots/{pn}_{sn}", dpi=100)
    plt.close()
    total_without_line = len(dlz + dgz)

    # Horizontal line so left and right doesn't make sense
    if total_without_line == 0:
        return (0,0)
    left_percent = (len(dlz) / total_without_line) * 100
    right_percent = (len(dgz) / total_without_line) * 100

    return (left_percent, right_percent)

def make_ecid_line(c_disc, emcon_line_points, disc_right_high_point):
    '''
    ecid = eminence-condyle intersect disc
    Anything related to Eminence-Condyle line intersecting with disc should be here.
    '''

    # Get rid of duplicates in c_disc but keep as list cause it's easier to work with
    c_disc = set(c_disc)
    c_disc = list(c_disc)
    
    # Find if emience-condyle line intersect with the disc
    intersect = has_common_data(c_disc, emcon_line_points) # function returns two items, [0] is Y/N for intersect, [1] is the list of common points

    # Find the length from disc_highest_point to where the eminence-condyle line intersects with the disc
    if intersect[0] == 'Y':
        ecid_point1, ecid_point2 = getShortestDistance([disc_right_high_point], intersect[1]) 
        ecid_line_length = get_distances_between_two_points(ecid_point1, ecid_point2)[2] # (x_dist,y_dist, eucl_dist)
    else:
        ecid_line_length = 0 # length is 0 if there is no intersection

    return ecid_line_length


def find_midpoint(alist):
    x_mid = round((alist[0][0] + alist[-1][0]) / 2)
    y_mid = round((alist[0][1] + alist[-1][1]) / 2)

    return (x_mid, y_mid)

def calc_auc(measurement):
    pass

leftovers = []

'''
Patients 6,7, and 15 have bad images. NO LABEL FOLDER FOR THESE PATIENTS
PATIENT 4 DOESN'T HAVE A FOLDER
'''
#Filename format Slice_1.json extract the slice number
slices = []
# Excel Data as a list of lists
Excel_Data = []
## Can we assume that all slices will have all three
def sort_patient(patient_folder):
    with open(os.path.join(pred_folder, patient_folder, 'source.txt')) as f:
        pattern = r"\((\d+)\)"
        patient_number = int(re.findall(pattern, f.read())[0])
        
            
    return patient_number

patient_folders = os.listdir(pred_folder)
print("Printting patient folder .....")
patient_folders = sorted(patient_folders, key=sort_patient)
for patient_folder in patient_folders: # Go through each patient's folder
    with open(os.path.join(pred_folder, patient_folder, 'source.txt')) as f:
        # 1.2.840.114202.4.4195027258.2508329886.3366014226.3210073009(250)/Label-1.2.840.114202.4.4195027258.2508329886.3366014226-Roxana
        pattern = r"\((\d+)\)"
        patient_number = int(re.findall(pattern, f.read())[0]) # IMPORTANT TO KEEP TRACK OF PATIENT'S NUMBER

    prediction = np.load(os.path.join(pred_folder, patient_folder, 'result.npz'))
    images, pred_masks, path = prediction["images"], prediction["masks"], prediction["paths"]
    bar = Bar("Preparing patient {} - ".format(patient_number), max=len(images))
    for slice_number, (image, pred_mask) in enumerate(zip(images, pred_masks)):
        bar.next()
        disc, condyle, eminence = pred_mask == 1, pred_mask == 2, pred_mask == 3
        # skip if no mask is predicted
        if all([disc.sum() > 4, condyle.sum() > 4, eminence.sum() > 4]):
            slices.append(slice_number)
            #print("Read contour points")
            #TODO  We could use Points instead
            disc_pts = np.array(disc.nonzero()).T[:, ::-1]
            condyle_pts = np.array(condyle.nonzero()).T[:, ::-1]
            eminence_pts = np.array(eminence.nonzero()).T[:, ::-1]
            c_disc = [tuple(pt) for pt in disc_pts]
            c_condyle  = [tuple(pt) for pt in condyle_pts]
            c_eminence = [tuple(pt) for pt in eminence_pts]

            #XY flip (set this as a command line argument)
            #c_disc     = [tuple([pt[1],pt[0]]) for pt in disc[0]['Contour']]
            #c_condyle  = [tuple([pt[1],pt[0]]) for pt in condyle[0]['Contour']]
            #c_eminence = [tuple([pt[1],pt[0]]) for pt in eminence[0]['Contour']]

            '''
            CONVENTION FOR ROUNDING: 
            ALL POINTS ARE ROUNDED
            MEASUREMENTS ARE NOT ROUNDED
            THIS IS BECAUSE THE ORIGINAL POINTS ARE ALL INTEGERS.
            '''

            # Find Center point of the Disc
            disc_centroid = get_mean_point(c_disc)


            # Construct the disc line and all associated calculations, as well as save the right-high point for ecid line
            disc_line_points, disc_length, top_right_best, bot_left_best = make_disc_line(c_disc) # ([point1, ..., pointn], disc_len)

            try:
                # Construct the Eminence-Condyle line and all associated calculations
                emcon_line_points, emcon_line_lengths = make_emcon_line(c_eminence, c_condyle,  bot_left_best) # ([point1, ..., pointn],[x_len, y_len, e_len])
            except:
                print("Error in making emcon line patient {} slice {}".format(patient_number, slice_number))
                continue
    
            # Calculate the disc ratio
            yellow_ratio = find_yellow_ratio(c_disc, emcon_line_points, patient_number, slice_number)

            # Eminence-Condyle intersect Disc line
            ecid_line_length = make_ecid_line(c_disc, emcon_line_points,top_right_best) 
            emcon_line_centroid = find_midpoint(emcon_line_points)

            # Find the distance between disc centroid to disc_best_fit line
            emcon_centroid_dist = get_distances_between_two_points(disc_centroid, emcon_line_centroid)

            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon(c_disc, outline=1, fill=1)
            ImageDraw.Draw(img).polygon(c_condyle, outline=2, fill=2)
            ImageDraw.Draw(img).line(c_eminence,  fill=3,width=2)
            ImageDraw.Draw(img).line(emcon_line_points, fill= 5) # Line p1->p2 GREEN
            ImageDraw.Draw(img).line((disc_centroid, emcon_line_centroid), fill = 4)
            # mask = np.array(img)
            # plt.imshow(mask)
            # plt.show()    
            
            image_name = f"./preds/mask_images_updated/Patient_{patient_number}_Slice_{slice_number}.png"
            plt.imsave(image_name, img)
            plt.close()
            # Add to Excel_Data list to be exported
            Excel_Data.append([patient_number, slice_number, yellow_ratio[0], yellow_ratio[1], emcon_centroid_dist[2]])
        
excel_path = "TMJ_Calc.xlsx"
if not os.path.exists(excel_path):
    workbook = openpyxl.Workbook()
else:
    workbook = openpyxl.load_workbook(excel_path, read_only=False)

#open workbook
sheet = workbook.active
row = 0
col = 0

print("len", len(Excel_Data))
num_folders = [item[0] for item in Excel_Data]
print(set(num_folders))
print(len(set(num_folders)))
excel_index = 2 # Start on row 2 for data values since row 1 is column name

for item in sorted(Excel_Data):
    print(item)
    sheet[f"A{excel_index}"] = item[0] # Patient Number
    sheet[f"B{excel_index}"] = item[1] # Slice Number
    sheet[f"C{excel_index}"] = item[2] # Percent Left Yellow
    sheet[f"D{excel_index}"] = item[3] # Percent Right Yellow
    sheet[f"E{excel_index}"] = item[4] # Disc Centroid to Emcon Line Euclidean Distance
    excel_index += 1

workbook.save(excel_path)
print('rocks')
