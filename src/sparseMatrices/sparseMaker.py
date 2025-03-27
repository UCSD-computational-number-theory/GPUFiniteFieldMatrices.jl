import numpy as np 
import os

class SparseMaker:

    v_str = np.vectorize(lambda x: str(int(x)))

    def __init__(self, N, P, E) -> None:
        # Size, Zp, and Entries
        self.N = N
        self.P = P
        self.E = E
        self.mat = np.zeros((self.N,self.N),dtype=np.uint)
        self.folder = ""

    def populate(self):
        self.mat = np.zeros((self.N,self.N),dtype=np.uint)

        # Choose ranom (rows, cols) to populate
        rows = np.random.randint(0,self.N,self.E,dtype=np.uint)
        cols = np.random.randint(0,self.N,self.E,dtype=np.uint)

        # Choose random values to populate
        vals = np.random.randint(1,self.P,self.E,dtype=np.uint)

        for i in range(self.E):
            self.mat[rows[i]][cols[i]] = vals[i]

    def print(self):
        print(self.mat)

    def export(self, *args):
        filename = f"{self.folder}sparseN{self.N}P{self.P}E{self.P}"
        filename += f"Num{args[0]}.csv" if len(args) != 0 else ".csv"
        self.populate()
        with open(filename, "w") as f:
            for i in range(self.N):
                f.write(",".join(self.v_str(self.mat[i])))
                f.write("\n")
    
    def multi_export(self, num, folder):
        self.folder = folder
        os.mkdir(self.folder)
        for index in range(num):
            self.export(index)

TestMaker = SparseMaker(220,121,100)
TestMaker.multi_export(10,"smallSize/")
