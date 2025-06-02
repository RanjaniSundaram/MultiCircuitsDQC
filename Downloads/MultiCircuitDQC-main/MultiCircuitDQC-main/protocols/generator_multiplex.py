'''http://www.dabeaz.com/generators/Generators.pdf
   page 101
'''
import threading, queue, time, random


def sendto_queue(source, thequeue):
    '''feed a generated sequece into a queue
    '''
    for item in source:
        thequeue.put(item)
    thequeue.put(StopIteration)

def genfrom_queue(thequeue):
    while True:
        item = thequeue.get()
        if item is StopIteration:
            break
        yield item

def gen_cat(sources):
    '''concatenate items from one or more source into a single sequences of items
    '''
    for src in sources:
        yield from src

class thread_safe_generator:
    def __init__(self, gen):
        self.gen = gen
        self.lock = threading.Lock()
    
    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.gen)  # resume to the yield expression in the generator

def simple_generator(id):
    i = 0
    while True:
        yield f'id={id},i={i}'
        i += 1
        time.sleep(random.random())

def multiplex(sources):
    '''
    Args:
        sources: list -- a list of generators
    Return:
        generator -- a single multiplexing generator of all the input generators
    '''
    in_q = queue.Queue()
    consumers = []
    for src in sources:
        t = threading.Thread(target=sendto_queue, args=(src, in_q))
        t.start()
        consumers.append(genfrom_queue(in_q))
    return gen_cat(consumers)


def main1():
    '''multiple endless generators running concurrently
    '''
    gens = []
    for i in range(3):
        gen = thread_safe_generator(simple_generator(i))
        gens.append(gen)

    multi = multiplex(gens)
    for i in multi:
        print(i)


if __name__ == '__main__':
    main1()
