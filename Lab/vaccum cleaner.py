class VacuumCleaner:
    def __init__(self, shape):
        self.shape = shape
        self.position = "A"
        self.running = False

    def start(self):
        self.running = True
        print(f"{self.shape} Vacuum Cleaner started at {self.position}")

    def stop(self):
        self.running = False
        print(f"{self.shape} Vacuum Cleaner stopped.")

    def left(self):
        if self.running:
            self.position = "A"
            print(f"{self.shape} moved to Left → Square A")

    def right(self):
        if self.running:
            self.position = "B"
            print(f"{self.shape} moved to Right → Square B")

    def dock(self):
        print(f"{self.shape} Vacuum Cleaner is docking for recharge.")

    def suck(self):
        if self.running:
            print(f"{self.shape} cleaned {self.position}")


shapes = ["Square", "Circle", "Triangle", "Hexagon"]

print("Choose Vacuum Cleaner Shape:")
for i, s in enumerate(shapes, 1):
    print(f"{i}. {s}")

choice = int(input("Enter choice (1-4): "))
shape = shapes[choice - 1]
vc = VacuumCleaner(shape)

print("\nCommands: start, stop, left, right, dock, suck, exit")

while True:
    cmd = input("Enter command: ").lower()
    if cmd == "start":
        vc.start()
    elif cmd == "stop":
        vc.stop()
    elif cmd == "left":
        vc.left()
    elif cmd == "right":
        vc.right()
    elif cmd == "dock":
        vc.dock()
    elif cmd == "suck":
        vc.suck()
    elif cmd == "exit":
        print("Exiting program.")
        break
    else:
        print("Invalid command. Try again.")
