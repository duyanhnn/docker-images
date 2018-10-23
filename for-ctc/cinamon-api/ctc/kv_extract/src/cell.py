#! /usr/bin/env python

class Cell:

    def __init__(self):
        self.id = None # contain id of cell
        self.value = None #contain value of cell
        self.next = None
        #relative position 
        self.top = None
        self.right = None
        self.bot = None
        self.left = None
        # variable to track 
        self.is_map = False
        self.loc = [0,0,0,0]

class CellMap:

    def __init__(self):
       self.current = None
       self.first = None
       self.len = 0
       self.cellList = None

    def add_cell(self,cell):
        if(self.first == None):
            self.first = cell
            self.current = cell
        else:
            self.next = cell
            self.current = self.next
        self.len += 1

    def get_cell(self):
        None

    def get_distance(self,cell1,cell2):
        None




# GRAB MAIN MODULE
if __name__ == '__main__':
   cell_map = CellMap()


