import numpy as np
import cv2
from math import sqrt
#import functools

WRAPPED_SIZE = 400
LOG = False

def dist(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


class Grid:
    #0: " "
    #1: "O"
    #2: "X"
    width, height = 10, 10

    def __init__(self, grid):
        self.grid = grid.copy()
        self.base = grid.copy()

        self.x = 0
        self.y = 0

    def is_valid(self):
        cell = self.grid[self.y, self.x]

        #If it's a O or a X
        if cell > 0:
            #countCol = list(self.grid[:,self.x]).count(cell)
            #countRow = list(self.grid[self.y]).count(cell)
            countCol = np.count_nonzero(self.grid[:,self.x] == cell)
            countRow = np.count_nonzero(self.grid[self.y] == cell)

            #If not too many in col nor in row
            if countCol <= 5 and countRow <= 5:
                # not -ooO--- / -xxX---
                beforeX2 = self.x < 2 or not (self.grid[self.y, self.x-2] == self.grid[self.y, self.x-1] == cell)
                beforeY2 = self.y < 2 or not (self.grid[self.y-2, self.x] == self.grid[self.y-1, self.x] == cell)

                # not --oOo-- / --xXx--
                sideX1 = self.x < 1 or self.x == self.width-1 or not (self.grid[self.y, self.x-1] == cell == self.grid[self.y, self.x+1])
                sideY1 = self.y < 1 or self.y == self.height-1 or not (self.grid[self.y-1, self.x] == cell == self.grid[self.y+1, self.x])

                # after ---Ooo- / ---Xxx-
                afterX2 = self.x > self.width-3 or not (self.grid[self.y, self.x+2] == self.grid[self.y, self.x+1] == cell)
                afterY2 = self.y > self.height-3 or not (self.grid[self.y+2, self.x] == self.grid[self.y+1, self.x] == cell)

                if all([beforeX2, beforeY2, sideX1, sideY1, afterX2, afterY2]):
                    return True



        return False

    def wrong(self):
        try:
            valueB, valueG = self.base[self.y, self.x], self.grid[self.y, self.x]
            #Can be modified
            if valueB == 0:
                if valueG < 2:
                    self.grid[self.y, self.x] += 1

                    return

                else:
                    self.grid[self.y, self.x] = 0

            #Move back
            if self.x > 0:
                self.x -= 1

            elif self.y > 0:
                self.y -= 1
                self.x = self.width-1

            else:
                raise Exception("No solution")

            self.wrong()

        except RecursionError as e:
            cv2.putText(img, "Recursion", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            return

        except:
            cv2.putText(img, "No solution", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            return

    def right(self):
        while True:
            if self.x == self.width-1 and self.y == self.height-1:
                return False

            self.x += 1
            self.x %= self.width
            if self.x == 0:
                self.y += 1

            if self.base[self.y, self.x] == 0:
                break

        return True

    def solve(self):
        valid = self.is_valid()

        if valid:
            if not self.right():
                return True

        else:
            self.wrong()


        return False


def decode(img_src):
    #Preprocess
    img = img_src.copy()
    #img = cv2.copyMakeBorder(img, 5,5,5,5, cv2.BORDER_CONSTANT, None, (0,0,0))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,2)
    thresh = np.invert(thresh)
    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1)))

    #cv2.drawContours(img_src, contours, -1, (0,255,0), 1)

    result = np.zeros([10,10])
    M = None

    #grid = thresh.copy()

    if not contours is None and len(contours) > 0:
        """
        count = 0
        max_ = -1

        maxPt = None
        #Find largest blob
        mask = np.zeros((grid.shape[0]+2,grid.shape[1]+2), np.uint8)

        for y in range(0,grid.shape[0]):
            for x in range(0,grid.shape[1]):
                if grid[y,x] == 255:
                    area = cv2.floodFill(grid, mask, (x,y), 100)[0]
                    if area > max_:
                        maxPt = (x,y)
                        max_ = area

        #Keep largest blob
        mask = np.zeros((grid.shape[0]+2,grid.shape[1]+2), np.uint8)
        cv2.floodFill(grid, mask, maxPt, 255)

        _, grid = cv2.threshold(grid, 200, 255, cv2.THRESH_BINARY)

        #grid = cv2.erode(grid, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
        """
        grid = np.zeros((thresh.shape[0]+2,thresh.shape[1]+2), np.uint8)

        """max_ = 0
        j = None
        for i in range(len(contours)):
            c, h = contours[i], hierarchy[0][i]
            area = cv2.contourArea(c)
            if area > max_:
                max_ = area
                j = i

        print(hierarchy[0][j])"""

        img_peri = grid.shape[0]*2+grid.shape[1]*2 - 500
        contours2 = list(filter(lambda c: cv2.arcLength(c, True) < img_peri, contours))

        if len(contours2) > 0:
            grid_border = max(contours2, key = cv2.contourArea)

            cx, cy = 0, 0
            for i in range(len(grid_border)):
                cx += grid_border[i][0][0]
                cy += grid_border[i][0][1]

            cx /= len(grid_border)
            cy /= len(grid_border)

            #Order contour clockwise
            #grid_border = np.array(list(sorted(grid_border, key = functools.cmp_to_key(lambda a,b: (a[0][0] - cx) * (b[0][1] - cy) - (b[0][0] - cx) * (a[0][1] - cy) ))))

            peri = cv2.arcLength(grid_border, True)
            grid_border = cv2.approxPolyDP(grid_border, 0.02*peri, True)
            cv2.drawContours(grid, [grid_border], -1, 255, 1)
            #cv2.drawContours(img_src, [grid_border], -1, (255,0,0), 1)

            """
            #Get contours of grid and cells
            contours2, hierarchy2 = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            approxs = []
            for cont in contours2:
                peri = cv2.arcLength(cont, True)
                approx = cv2.approxPolyDP(cont, 0.02*peri, True)
                approxs.append(approx)

            cv2.drawContours(img_src, approxs, 0, (0,0,255), 1)
            """


            #if len(approxs) == 101:
            #Check that grid_border is a quadrilateral
            if len(grid_border) == 4:
                from_ = np.array(list(map(lambda _: _[0], grid_border)),dtype="float32")
                to_ = np.array([[0,0],[WRAPPED_SIZE,0],[WRAPPED_SIZE,WRAPPED_SIZE],[0,WRAPPED_SIZE]],dtype="float32")

                #Warp image to cancel perspective
                M = cv2.getPerspectiveTransform(from_, to_)
                warped = cv2.warpPerspective(thresh, M, (WRAPPED_SIZE,WRAPPED_SIZE))

                w, h = int(warped.shape[1]/10),int(warped.shape[0]/10)

                #warped = cv2.GaussianBlur(warped, (3,3), 0)
                #warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,2)
                _, warped = cv2.threshold(warped, 200, 255, cv2.THRESH_BINARY)
                
                #warped = cv2.erode(warped, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
                #warped = cv2.dilate(warped, cv2.getStructuringElement(cv2.MORPH_CROSS, (4,4)))

                overlay = np.zeros((warped.shape[0],warped.shape[1],3), dtype=np.uint8)
                m = 10
                m2 = int(min(w, h)/2-3)
                
                for y in range(10):
                    for x in range(10):
                        mask = np.zeros(warped.shape, dtype=np.uint8)
                        
                        mask[y*h+m:y*h+h-m, x*w+m:x*w+w-m] = 255
                        #mask[y*h:y*h+h, x*w:x*w+w] = 255

                        _ = cv2.bitwise_and(warped, mask)
                        _[mask==0] = 255

                        cts, hrcy = cv2.findContours(_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                        #cts, hrcy = cv2.findContours(_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        X, Y = x*w, y*h
                        #margin = 10
                        
                        #cv2.putText(overlay, str(len(cts)), (X,Y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        
                        empty = cv2.mean(warped[y*h+m:y*h+h-m, x*w+m:x*w+w-m])[0] < 10
                        cross = cv2.mean(warped[y*h+m2:y*h+h-m2, x*w+m2:x*w+w-m2])[0] > 128
                        
                        if not empty:
                            if cross:
                                result[y,x] = 2
                                #cv2.line(img_src, (X+margin,Y+margin), (X+w-margin,Y+h-margin), (0,120,255), 2)
                                #cv2.line(img_src, (X+margin,Y+h-margin), (X+w-margin,Y+margin), (0,120,255), 2)
                                cv2.rectangle(overlay, (X,Y), (X+w,Y+h), (0,255,0), -1)

                            else:
                                result[y,x] = 1
                                #cv2.circle(img_src, (int(X+w/2), int(Y+h/2)), int(min(w,h)/2)-margin, (0,120,255), 2)
                                cv2.rectangle(overlay, (X,Y), (X+w,Y+h), (0,0,255), -1)

                        #_ = cv2.cvtColor(_, cv2.COLOR_GRAY2BGR)

                        #cv2.drawContours(overlay, cts, -1, (0,255,255),2)

                        #cv2.imshow('cells', _)

                #cv2.imshow('overlay', overlay)
                overlay = cv2.warpPerspective(overlay, M, (img_src.shape[1], img_src.shape[0]), None, cv2.WARP_INVERSE_MAP)

                gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                cv2.addWeighted(img_src, 1, overlay, 0.2, 0, img_src)

                cv2.drawContours(img_src, [grid_border], -1, (255,0,0), 1)

                cv2.imshow('warped', warped)
            #cv2.imshow('grid', grid)

    #cv2.imshow('img', img)
    cv2.imshow('thresh', thresh)

    #cv2.imshow('gray', imgray)

    return [result, M]

def encode(grid, M, shape):
    img = np.zeros((WRAPPED_SIZE,WRAPPED_SIZE,3), dtype=np.uint8)

    #M = np.linalg.pinv(M)

    w, h = int(WRAPPED_SIZE/10), int(WRAPPED_SIZE/10)

    margin = 10
    colour = (255,255,255)

    for y in range(10):
        for x in range(10):
            X, Y = x*w, y*h

            cell = grid[y,x]

            if cell == 1:
                cv2.circle(img, (int(X+w/2), int(Y+h/2)), int(min(w,h)/2)-margin, colour, 2)

            elif cell == 2:
                cv2.line(img, (X+margin,Y+margin), (X+w-margin,Y+h-margin), colour, 2)
                cv2.line(img, (X+margin,Y+h-margin), (X+w-margin,Y+margin), colour, 2)

    for i in range(11):
        cv2.line(img, (i*w, 0), (i*w, shape[1]), (235,204,75), 2)
        cv2.line(img, (0, i*h), (shape[0], i*h), (235,204,75), 2)

    result = cv2.warpPerspective(img, M, (shape[1], shape[0]), None, cv2.WARP_INVERSE_MAP)

    return result

#cv2.waitKey()

if __name__ == "__main__":
    #original = cv2.imread('binoxxo_1.jpg')
    #img = original.copy()
    #print(decode(img))

    avg_tmp = []

    cam = cv2.VideoCapture(0)

    while True:
        ret_val, original = cam.read()
        img = original.copy()

        #img = cv2.flip(img, 1)

        grid, M = decode(img)
        #cv2.imshow('Webcam', original)


        if not M is None and np.count_nonzero(grid) > 5:
            avg_tmp.append(grid)

            if len(avg_tmp) > 10:
                avg_tmp.pop(0)


            avg = np.round(np.average(avg_tmp,0))

            grid = Grid(avg)

            count = 0
            while True:
                solved = grid.solve()
                count += 1

                if solved or count > 500:
                    if count > 500:
                        cv2.putText(img, "Max count", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    break

            if solved:
                solved_img = encode(grid.grid-grid.base, M, img.shape)

                tmp = cv2.cvtColor(solved_img, cv2.COLOR_BGR2GRAY)
                _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                encoded = img.copy()

                """
                b, g, r = cv2.split(solved_img)
                rgba = [b,g,r, alpha]
                dst = cv2.merge(rgba,4)

                alpha_s = dst[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    encoded[:, :, c] = (alpha_s * dst[:, :, c] +
                                              alpha_l * encoded[:, :, c])
                """


                cv2.copyTo(solved_img, alpha, encoded)

                cv2.imshow('Scanner', encoded)

            else:
                cv2.putText(img, "Unsolved", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.imshow('Scanner', img)

        else:
            cv2.putText(img, "No grid", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imshow('Scanner', img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

    """
    grid, M = decode(img)
    grid = Grid(grid)
    cv2.imshow('Scanner', img)
    cv2.waitKey()

    while True:
        solved = grid.solve()

        if solved:
            break

    solved_img = encode(grid.grid, M, img.shape)
    #solved_img = cv2.cvtColor(solved_img, cv2.COLOR_BGR2BGRA)

    tmp = cv2.cvtColor(solved_img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(solved_img)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)

    alpha_s = dst[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    encoded = original.copy()

    for c in range(0, 3):
        encoded[:, :, c] = (alpha_s * dst[:, :, c] +
                                  alpha_l * encoded[:, :, c])

    cv2.imshow("Solved", encoded)

    print(grid.grid)
    cv2.waitKey()"""
