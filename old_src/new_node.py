class newnode:
    def __init__(self, data, value):
        self.key = data
        self.integral_value = value
        self.left = None
        self.right = None
    def __str__(self):
        s = ""
        if self.left:
            s += str(self.left) + " "
        s+= str(self.key) + " "
        if self.right:
            s+= str(self.right) + " "
        return s