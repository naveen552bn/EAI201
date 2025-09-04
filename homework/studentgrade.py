def grade_student(subject_marks):
    total = sum(subject_marks.values())
    avg = total / len(subject_marks)

    if avg >= 90:
        grade = "A+"
    elif avg >= 80:
        grade = "A"
    elif avg >= 70:
        grade = "B"
    elif avg >= 60:
        grade = "C"
    elif avg >= 50:
        grade = "D"
    else:
        grade = "F"

    return {"Total": total, "Average": avg, "Grade": grade}



marks = {}
n = int(input("Enter number of subjects: "))

for i in range(n):
    subject = input(f"Enter subject {i+1} name: ")
    score = float(input(f"Enter marks for {subject}: "))
    marks[subject] = score

result = grade_student(marks)
print("\nResult:", result)
