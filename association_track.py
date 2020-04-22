import numpy as np
from PIL import Image
from mrcnn import visualize as vz
import cv2
import statistics as stats

# FILE CONTAINING FUNCTIONS USED TO TRACK VIA MASK ASSOCIATION
# AUTHOR - John Allen QMUL


# generate initial object representations and save to dictionary according to colours
def initialise_representation(image, masks, colors, ids):

    representation_dict = create_representation_dict(image, masks, colors, ids)

    representation_store = [representation_dict]

    return representation_store


# create representation dictionary
def create_representation_dict(image, masks, colors, ids):

    representation_dict = {}

    for i in range(len(ids)):

        colorID = np.array(colors[i])
        colorID = np.append(colorID, ids[i])

        rep = mask_to_representation(masks[:, :, i], image)
        #cv2.imwrite('%05d.jpg' % i, np.uint8(np.true_divide(rep, 3)))
        representation_dict.update({tuple(colorID): rep})

    return representation_dict


# determine best colour based on representations
def determine_best_colors(image, masks, ids, rep_store, adjacent=False):

    colors = np.zeros((len(ids), 3))
    sort_colors = np.zeros((len(ids), 4))
    bool_colors = [False for i in range(len(ids))]
    count = 0

    # Continue until all objects have colors
    while sum(bool_colors) < len(ids):

        # New Object
        if count > (len(ids)+1):

            for c in range(len(ids)):
                if bool_colors[c] is False:
                    bool_colors[c] = True
                    colors[c] = vz.generate_new_random_color(colors)

        # Existing Object
        else:

            # Iterate through all detected objects
            for N in range(len(ids)):

                if bool_colors[N] == False:

                    # Iterate through all previous frames
                    for k in range(len(rep_store)):

                        # initialise best error
                        best_error = 999999

                        # Iterate through all detections for frame
                        for colorID in rep_store[k]:

                            # check class id of detection matches detection in previous frame
                            if ids[N] == colorID[3] and vz.is_color_unique(np.uint8(np.multiply(np.array(colorID[0:3]), 255)),
                                                                           np.uint8(np.multiply(colors, 255))):

                                # generate representation of detection to assign color
                                rep = mask_to_representation(masks[:, :, N], image)
                                error = calculate_error(rep, rep_store[k][colorID]) # calculate error between detection reps

                                # assign new best error and color if error is lower than previous best error
                                if error < best_error:

                                    best_error = error
                                    best_color = tuple(colorID[0:3])

                        if adjacent:

                            return best_color

                        # convert best color to list and insert into list of best colors and errors for each previous frame
                        list_best_color = list(best_color)
                        list_best_color.append(best_error)

                        if k > 0:

                            array_color_error = np.append(array_color_error, np.array([list_best_color]), axis=0)

                        else:

                            array_color_error = np.array([list_best_color])

                # determine the mode color across all frames and the lowest error for this color
                color_error = find_mode_color_min_error(array_color_error)

                # insert the mode color and error into sort colors
                sort_colors[N] = color_error

            errors = sort_colors[:, 3]
            prop_colors_scaled = sort_colors[:, 0:3]
            prop_colors = np.uint8(np.multiply(sort_colors[:, 0:3], 255))

            for a in range(np.shape(prop_colors)[0]):

                A = prop_colors[a, :]

                if bool_colors[a] is True or not vz.is_color_unique(A, colors):
                    continue

                #duplicates = np.where(prop_colors[:, 0] == A[0] and prop_colors[:, 1] == A[1] and prop_colors[:, 2] == A[2], errors[:], 9999999)

                duplicates = []
                for b in range(np.shape(prop_colors)[0]):
                    if prop_colors[b, 0] == A[0] and prop_colors[b, 1] == A[1] and prop_colors[b, 2] == A[2]:
                        duplicates.append(errors[b])

                #if len(duplicates) > 1:

                #duplicate_errors = [errors[i] for i in duplicates]
                min_duplicate = min(duplicates)
                color_index = np.where(errors == min_duplicate)[0][0]
                bool_colors[color_index] = True
                colors[color_index, :] = prop_colors_scaled[color_index, :]

                '''else:
    
                    bool_colors[a] = True
                    colors[a] = list(prop_colors_scaled[a])'''
        count += 1

    return colors


# extracts individual object elements from image and removes all other pixels to represent object
def mask_to_representation(mask, image):

    new_image = np.sum(image, axis=2)

    object_image = np.where(mask == True, new_image, 0)

    indices = np.where(mask == True)

    object_rep = object_image[min(indices[0]): max(indices[0])+1, min(indices[1]): max(indices[1]+1)]

    return object_rep


# rescales stored representation to match size of new representation
def scale_representations(repA, repB):

    if np.size(repA) == np.size(repB):

        if np.shape(repA)[0] != np.shape(repB)[0]:

            height, width = np.shape(repA)

            imgB = Image.fromarray(np.uint8(np.true_divide(repB, 3)))
            newB = np.array(imgB.resize((width, height), Image.BICUBIC))
            newB = np.multiply(np.uint64(newB), 3)

            return repA, newB

        else:
            return repA, repB

    elif np.size(repA) > np.size(repB):

        height, width = np.shape(repB)

        imgA = Image.fromarray(np.uint8(np.true_divide(repA, 3)))
        newA = np.array(imgA.resize((width, height), Image.BICUBIC))
        newA = np.multiply(np.uint64(newA), 3)

        return newA, repB

    elif np.size(repA) < np.size(repB):

        height, width = np.shape(repA)

        imgB = Image.fromarray(np.uint8(np.true_divide(repB, 3)))
        newB = np.array(imgB.resize((width, height), Image.BICUBIC))
        newB = np.multiply(np.uint64(newB), 3)

        return repA, newB


# determine intersection of scaled object representations and a coefficient used in error calculation
# if intersection is high compared to union coefficient is low
def representation_intersection(repA, repB):

    A, B = scale_representations(repA, repB)

    intersection = np.where((A > 0) & (B > 0), True, False)

    union = np.where((A > 0) | (B > 0), True, False)

    coefficient = np.sum(union) / np.sum(intersection)

    return intersection, coefficient, A, B


# calculate pixel wise error for each colour channel of representations using intersection and coefficient
def calculate_error(repA, repB):

    intersec, coef, A, B = representation_intersection(repA, repB)

    intersecA = np.int64(np.where(intersec == True, A, 0))
    intersecB = np.int64(np.where(intersec == True, B, 0))

    mean_color_pixel_error = (coef) * (abs(np.sum(np.subtract(intersecA, intersecB))) / (np.sum(intersec)*3))

    return mean_color_pixel_error


# Finds mode color in set of proposed colors
def find_mode_color_min_error(frame_color):

    frequency_dict = {}

    for c in range(np.shape(frame_color)[0]):

        color = frame_color[c, :]

        try:

            frequency_dict.update({tuple(color[0:3]): 0})

        except:

            frequency_dict[tuple(color[0:3])] += 1

    mode_color = max(zip(frequency_dict.keys()))[0]

    array_mode = np.uint8(np.multiply(np.array(mode_color), 255))
    fc = np.uint8(np.multiply(frame_color[:, 0:3], 255))

    #errors = np.where(fc[:, 0] == array_mode[0] and fc[:, 1] == array_mode[1] and fc[:, 2] == array_mode[2], frame_color[:, 3], 9999999)

    errors = []
    for i in range(np.shape(fc)[0]):
        if fc[i, 0] == array_mode[0] and fc[i, 1] == array_mode[1] and fc[i, 2] == array_mode[2]:
            errors.append(frame_color[i, 3])


    error = min(errors)
    #error = stats.mean(errors)


    mode_color_error = list(mode_color)

    mode_color_error.append(error)

    return mode_color_error
