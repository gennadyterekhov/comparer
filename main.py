import lsa
import sys


def get_original_filename():
    return sys.argv[1:2][0]


def get_student_filenames():
    return sys.argv[2:]


def get_contents(filename):
    input_file = open(filename, 'r', encoding='utf8')
    temp = input_file.read()
    input_file.close()
    return temp


def get_contents_from_filenames(filenames):
    contents = []
    for f in filenames:
        input_file = open(f, 'r', encoding='utf8')
        temp = input_file.read()
        contents.append(temp)
        input_file.close()
    return contents


def get_bool_from_coef(coef):
    if coef > 0.5:
        return True
    return False


if __name__ == '__main__':
    original = get_contents(get_original_filename())
    students = get_contents_from_filenames(get_student_filenames())

    result = lsa.do_all(original, students)

    for res in result:
        print('работа ученика {}: {}'.format(res[0], get_bool_from_coef(res[1])))
