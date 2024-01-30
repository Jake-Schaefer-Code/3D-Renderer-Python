import numpy as np
import random
import math
import pygame
import time

WIDTH2 = 512
HEIGHT2 = 512

class grid:
    def __init__(self, grid):
        self.grid = grid

def normalize(vec, tolerance=0.00001):
    mag2 = sum(n * n for n in vec)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
    return vec/mag

def random_vector(cellcorner):
    x = np.dot(cellcorner, [123.4,234.5])
    y = np.dot(cellcorner, [234.5,345.6])
    v = np.array([x,y])
    v = np.sin(v)
    v = v * 143758
    t = time.time()
    v = np.sin(v + t)
    return v

def returncell(point, cellcorners):
    #print(point)
    point = point/WIDTH2*8-1
    #print(point)
    index = np.ceil(point)
    index = index.astype(int)
    #print(index)
    #print(cellcorners[index[0]][index[1]])
    return cellcorners[index[0]][index[1]]

def vecs2point(point, cell, frac):
    vectors = -cell + point
    #print(cell,point) 
    #print(vectors)
    return vectors
    
def dots(point, cell, frac):
    vectors = vecs2point(point, cell,frac)/WIDTH2*8
    randvecs = np.array([random_vector(corner) for corner in cell])/WIDTH2*8
    #print(randvecs)
    tx, ty = point/WIDTH2*8 - np.floor(point/WIDTH2*8).astype(int)
    #print(tx,ty)
    dottl = np.dot(randvecs[0],vectors[0])
    dottr = np.dot(randvecs[1],vectors[1])
    xt = lerp(dottl,dottr,tx)

    dotbr = np.dot(randvecs[2],vectors[2])
    dotbl = np.dot(randvecs[3],vectors[3])
    xb = lerp(dotbl,dotbr, tx)
    
    val = lerp(xt,xb,ty)
    #print(np.abs(val*100))

    #dots = np.array([dottl,dottr,dotbr,dotbl])
    return np.abs(val*100)


def lerp(v0,v1,t):
    return v0 + t*(v1-v0)

def generate_noise(size):
    grid = np.array([np.array([x,y]) for x in range(8) for y in range(8)]) * WIDTH2/8
    cellcorners = np.array([[np.array([[0,0],[0,0],[0,0],[0,0]]) for _ in range(8)] for _ in range(8)])
    for i in range(8):
        for j in range(8):
            tl = grid[i*8+j]
            tr = grid[i*8+j] + np.array([1,0])* WIDTH2/8
            bl = grid[i*8+j] + np.array([0,1])* WIDTH2/8
            br = grid[i*8+j] + np.array([1,1])* WIDTH2/8
            cellcorners[i][j] = np.array([tl,tr,br,bl])
    cellcorners = cellcorners 
    numcell = size
    map = np.zeros([numcell,numcell])
    for i in range(numcell):
        for j in range(numcell):
            point = np.array([i*WIDTH2/numcell+WIDTH2/(2*numcell),j*HEIGHT2/numcell + WIDTH2/(2*numcell)])
            cell = returncell(point, cellcorners)
            colorval = dots(point,cell, WIDTH2/(2*numcell))#*225
            map[i][j] = colorval
    print(map)
    return map

def main2():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH2, HEIGHT2))
    running = True
    clock = pygame.time.Clock()
    grid = np.array([np.array([x,y]) for x in range(8) for y in range(8)]) * WIDTH2/8
    cellcorners = np.array([[np.array([[0,0],[0,0],[0,0],[0,0]]) for _ in range(8)] for _ in range(8)])
    for i in range(8):
        for j in range(8):
            tl = grid[i*8+j]
            tr = grid[i*8+j] + np.array([1,0])* WIDTH2/8
            bl = grid[i*8+j] + np.array([0,1])* WIDTH2/8
            br = grid[i*8+j] + np.array([1,1])* WIDTH2/8
            cellcorners[i][j] = np.array([tl,tr,br,bl])
    cellcorners = cellcorners 
    """
    j = 0       j = 1
    0 1 9 10   |  9 10 18 19
    1 2 10 11  |  
    2 3 11 12  | """
    
    numcell =64
    map = np.zeros([numcell,numcell])
    for i in range(numcell):
        for j in range(numcell):
            point = np.array([i*WIDTH2/numcell+WIDTH2/(2*numcell),j*HEIGHT2/numcell + WIDTH2/(2*numcell)])
            cell = returncell(point, cellcorners)
            colorval = dots(point,cell, WIDTH2/(2*numcell))*225
            map[i][j] = colorval
            #print(colorval)
            #pygame.draw.circle(screen, "white", point, 2)
    print(map)
    print()
    generate_noise(64)
    

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        

        screen.fill("black")
        
        """x = random.uniform(0,1)
        y = random.uniform(0,1)
        point = np.array([x*720,y*720])
        cell = vecs2point(point,cellcorners)
        pygame.draw.polygon(screen, "red", cell, 1)
        pygame.draw.circle(screen,"red", point, 5)"""
        numcell =64
        map = np.zeros([numcell,numcell])
        
        for i in range(numcell):
            for j in range(numcell):
                
                
                point = np.array([i*WIDTH2/numcell+WIDTH2/(2*numcell),j*HEIGHT2/numcell + WIDTH2/(2*numcell)])
                cell = returncell(point, cellcorners)
                colorval = dots(point,cell, WIDTH2/(2*numcell))*225
                map[i][j] = colorval
                #print(colorval)
                rect = pygame.Rect([i*WIDTH2/numcell,j*HEIGHT2/numcell],[WIDTH2/numcell,HEIGHT2/numcell])
                pygame.draw.rect(screen, (colorval,colorval,colorval),rect)
                #pygame.draw.circle(screen, "white", point, 2)
        #print(map)
        #print()
        #point = np.array([45,45])
        #vectors = vecs2point(point,cellcorners)
        #pygame.draw.circle(screen,"red", point, 5)

        #pygame.draw.line(screen, "red", cellcorners[0][0].astype(int)+vectors[0].astype(int), cellcorners[0][0].astype(int))
        #pygame.draw.line(screen, "red", cellcorners[0][1].astype(int)+vectors[1].astype(int), cellcorners[0][1].astype(int)) 
        for point in grid:
            vector = random_vector(point)*WIDTH2/16
            pygame.draw.line(screen, "white", point, (point+vector))
            #pygame.draw.circle(screen,"black",point,2)


        pygame.display.flip()
        #clock.tick(60)

if __name__ == "__main__":
    main2()

    