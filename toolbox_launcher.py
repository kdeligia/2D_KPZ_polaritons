from numpy import *
import pickle
def get_fname(I, prefix=''):
	a = list(I.keys())
	k = argsort(a)
	fname = ''
	for s in k:
		if isinstance(I[a[s]], str):
			fname += a[s] + '_' + I[a[s]] + '_'
		elif isinstance(I[a[s]], float) or isinstance(I[a[s]], int):
			fname += a[s] + '_{:g}'.format(I[a[s]]) + '_'
	return prefix + fname[:len(fname) - 1]

def get_input_file(I, prefix = ''):
    fname = get_fname(I,prefix)+'.txt'
    f = open(fname,'w')
    f.write('input\n')
    f.write('{\n')
    a = list(I.keys())
    k = argsort(a)
    for s in k:
        f.write(a[s]+ ' = {:g}\n'.format(I[a[s]]))
    f.write('}\n')
    f.close()
    return fname

def get_input_file_python(I, prefix = ''):
    fname = get_fname(I,prefix)+'.inp'
    f = open(fname,'wb')
    pickle.dump(I,f)
    f.close()
    return fname


def get_Ilist(I, i=0):
	a = sort(list(I.keys()))
	N = len(a)
	if i == N:
		return [I]
	elif isinstance(I[a[i]], list):
		L = []
		for s in I[a[i]]:
			I1 = I.copy()
			I1[a[i]] = s
			Il = get_Ilist(I1, i=i + 1)
			L += Il
		return L
	else:
		return get_Ilist(I, i + 1)
