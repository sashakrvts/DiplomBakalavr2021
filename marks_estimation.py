def calculate_mark(blank_answers, right_answers):
    answers_amount = len(right_answers)
    mark = 100
    for i in range(answers_amount):
        if blank_answers[i+1] != right_answers[i]:
            print(blank_answers[i+1],right_answers[i])
            mark -= 100/answers_amount
            print(mark)
    if (mark < 0):
        mark = 0
    return mark