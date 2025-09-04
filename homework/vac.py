def vacuum_square(length):
    area_room = length ** 2
    area_cover = area_room      
    gap = area_room - area_cover
    print("\n--- Square Vacuum ---")
    print("Room:", area_room)
    print("Covered:", area_cover)
    print("Uncovered:", gap)
    print("Efficiency:", (area_cover / area_room) * 100, "%")

def vacuum_circle(length):
    area_room = length ** 2
    radius = length / 2
    area_cover = 3.1416 * radius ** 2
    gap = area_room - area_cover
    print("\n--- Circle Vacuum ---")
    print("Room:", area_room)
    print("Covered:", area_cover)
    print("Uncovered:", gap)
    print("Efficiency:", (area_cover / area_room) * 100, "%")

def vacuum_triangle(length):
    area_room = length ** 2
    area_cover = (1.73205 / 4) * (length ** 2)   
    gap = area_room - area_cover
    print("\n--- Triangle Vacuum ---")
    print("Room:", area_room)
    print("Covered:", area_cover)
    print("Uncovered:", gap)
    print("Efficiency:", (area_cover / area_room) * 100, "%")

print("Vacuum Options:\n A. Square\n B. Circle\n C. Triangle")
choice = input("Choose (A/B/C): ").strip().upper()
length = float(input("Enter the side of the room: "))

if choice == "A":
    vacuum_square(length)
elif choice == "B":
    vacuum_circle(length)
elif choice == "C":
    vacuum_triangle(length)
else:
    print("Invalid option!")
