import sys
import wiki_graph

# print(cosine_similarity.item().__round__(3))


def main(argv):
    subject = ""
    cor_index = 1
    for i in range(len(argv)):
        try:
            float(argv[i])
            cor_index = i  # save index of correlation limit
            break
        except ValueError:
            subject += argv[i] + " "
    subject += "\b"  # delete the last space
    print("Searching in wikipedia for: " + subject)
    if subject == '' or float(argv[cor_index]) < 0 or float(argv[cor_index]) >= 1 or argv[cor_index + 1].isdigit() == 0 \
            or int(argv[cor_index + 1]) < 1:
        # arguments over index cor_index + 1 are neglected
        # correlation limit is the lower limit for the correlation metric which is in the range [0,1]
        # isdigit() returns false if tree depth is decimal
        print("Usage is: python3 ./main.py <subject: string> <correlation limit: float> <tree depth: int>")
        exit()
    wiki_graph.wiki_graph(subject, float(argv[cor_index]), int(argv[cor_index + 1]))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage is: python3 ./main.py <subject> <correlation limit> <tree depth>")
        exit()
    main(sys.argv[1:])  # don't pass the program call to the main function
