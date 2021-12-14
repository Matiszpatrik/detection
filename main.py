import numpy as np
import cv2
import matplotlib.pylab as plt
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines (img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            #print('x1', x1, 'x2', x2, 'y1', y1, 'y2', y2)
            cv2.line(blank_image, (x1, y1), (x2, y2), (0,255,0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def calculate_angle(lines):
    counter = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            diff = abs(y1 - y2)
            if diff < 200:   #biztosítja, hogy csak futófelületen lévő egyenest vizsgáljunk (a kerék oldalát ne)

                if counter == 0:
                    vector_x = x2 - x1
                    vector_y = y2 - y1
                    normal_vector_x = vector_y
                    normal_vector_y = -1 * vector_x
                    result_of_equation = normal_vector_x * x1 + normal_vector_y * y1
                    print(normal_vector_x, 'x', '+', normal_vector_y, 'y', '=', result_of_equation)
                    counter = 1
                elif counter == 1:
                    vector_x2 = x2 - x1
                    vector_y2 = y2 - y1
                    normal_vector_x2 = vector_y2
                    normal_vector_y2 = -1 * vector_x2
                    result_of_equation2 = normal_vector_x2 * x1 + normal_vector_y2 * y1
                    print(normal_vector_x2, 'x', '+', normal_vector_y2, 'y', '=', result_of_equation2)
    #print(normal_vector_x, normal_vector_y, normal_vector_x2, normal_vector_y2)
    scalar_product = (normal_vector_x2 * normal_vector_x) + (normal_vector_y2 * normal_vector_y)
    length_of_normal_vector = math.sqrt(normal_vector_x * normal_vector_x + normal_vector_y * normal_vector_y)
    length_of_normal_vector2 = math.sqrt(normal_vector_x2 * normal_vector_x2 + normal_vector_y2 * normal_vector_y2)
    angle_radian = math.acos((scalar_product)/((length_of_normal_vector)*(length_of_normal_vector2)))
    angle_degree = math.degrees(angle_radian)
    print('Bezárt szög:', format(angle_degree, ".2f"))


image = cv2.imread('car_2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (1450, height),
    (1450, 0),
    (1900, 0),
    (1900, height)
]

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 120)
cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

lines = cv2.HoughLinesP(cropped_image,
                        rho=2,
                        theta=np.pi/60,
                        threshold=40,
                        lines=np.array([]),
                        minLineLength=250,
                        maxLineGap=45)

image_with_lines = draw_the_lines(image, lines)
calculate_angle(lines)

plt.imshow(image_with_lines)
plt.show()

