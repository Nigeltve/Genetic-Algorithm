"""
This script uses the concepts of Genetic Algorithms to be able to get the correct sentence.

Doesnt use and Genetic algorithm packages (to understand the concept of how it works)

by: Nigel Thornewill von Essen
"""
import numpy as np
import string

# np.random.seed(17)

correct_sentence = 'cores'
pop_size = 200
mutation_rate = 15  # 0 is no mutation, 100 is definitely going to mutate.


def fit_func(correct: str, given: str):
    """
    Scores a entity based on how close it is to the target sentence/word

    :param correct: This is the correct sentence the Population is going to try and get.
    :param given: This is what the specimen came up with

    :return:  what the score is for the specimen int: score
    """

    raw_score = 0
    len_correct = len(correct)

    for letter_1, letter_2 in zip(correct, given):
        if letter_1 == letter_2:
            raw_score += 1

    # normalized score 0 is the worst 100 is the best.
    score = round((raw_score / len_correct) * 100, 3)
    return score


def Population_generator(size: int):
    """
    Generates a population based on the length of the sentence/word its trying to get

    :param size: This is the size that the
    :return: population as a set
    """
    choice_set = list(string.ascii_letters + string.digits)
    population = list()

    for pop_size in range(size):
        entity = ''
        for char in range(len(correct_sentence)):
            entity = entity + np.random.choice(choice_set)

        population.append(entity)

    return population


def reproduction(reproduction_list: list):
    """
    takes a list that is ready to be reporduced takes pairs and uses them for reproduction

    :param reproduction_list: ready list for reproduction
    :return: new generation
    """

    corssed_pop = []
    if len(reproduction_list) % 2 == 0:
        for i in range(0, len(reproduction_list), 2):
            random_split = 3
            p1_fh = reproduction_list[i][:random_split]
            p1_sh = reproduction_list[i][random_split:]

            p2_fh = reproduction_list[i + 1][:random_split]
            p2_sh = reproduction_list[i + 1][random_split:]

            new_child_1 = p1_fh + p2_sh
            new_child_2 = p1_sh + p2_fh
            corssed_pop.append(new_child_1)
            corssed_pop.append(new_child_2)
        return corssed_pop

    elif len(reproduction_list) % 2 == 1:
        print('The population is not even, please make it even')


def selection(scored_pop: list):

    # will choose based off their scores (which are percentages)

    weight_list = []
    choice_list = []
    for entity, weight in scored_pop:

        choice_list.append(entity)
        weight_list.append(weight)

    # need to normalise weight list so it sum to 1
    max = sum(weight_list)
    for idx, ele in enumerate(weight_list):
        # can get errors with smaller groups as its unlikely that it will get a letter on the first go
        percent = (ele / max)
        weight_list[idx] = percent

    # selects what is chosen
    choice = np.random.choice(choice_list, pop_size, True, weight_list)  # selects who breads

    return choice


def mutation(crossed_pop: list):
    choice_set = list(string.ascii_letters + string.digits)
    mutate_list = []

    for entity in crossed_pop:
        entity = list(entity)
        for idx, char in enumerate(entity):
            num = np.random.randint(0, 101)
            if num < mutation_rate:
                entity[idx] = np.random.choice(list(choice_set))
        entity = "".join(entity)
        mutate_list.append(entity)

    return mutate_list


def Genetric_Algorithm(population: list):
    """
    This fucntion will take in a population and outputs an "evolved" population

    :param population: the initial population to work with.

    :return: evolved population
    """
    new_population = []
    init_pop = population
    # give each entity a score

    entity_score = list()
    top_score = 0
    top_entity = ''
    generation_number = 0

    while top_score < 100:
        generation_number += 1
        print('\nGENERATION NUMBER: ', generation_number)

        for entity in init_pop:
            scoreing = (entity, fit_func(correct_sentence, entity))
            entity_score.append(scoreing)

        print(max(entity_score, key=lambda x: x[1]))
        # choose who gets picked.
        # takes the n-1 of the population and reproduces with them
        entity_score = sorted(entity_score, key=lambda x: x[1], reverse=True)

        # need to find top score
        top_entity, top_score = max(entity_score, key=lambda x: x[1])

        if top_score != 100:
            # selects who is bread by probability based on fitness score. better means higher probability.
            selected = selection(entity_score)

            # reproduction by crossover method
            reproducted = reproduction(selected)

            # mutate the new population

            new_population = mutation(reproducted)
            init_pop = new_population

            # print(new_population)

    print('Has now got the sentence you are looking for with a score of {} word was {}'.format(
        top_score, top_entity))

    return top_score, top_entity


print('the objective is to get the following string right:\n---->\t {} \t<----'.format(correct_sentence))
population = Population_generator(pop_size)  # sets the initial population
Genetric_Algorithm(population)
