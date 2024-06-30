def dict_update_or_default(d: dict, key, default, update_f) -> dict:
	vOpt = d.get(key)
	if vOpt is None:
		d[key] = default
	else:
		d[key] = update_f(vOpt)
	return d


def partition(l: list, f) -> (list, list):
	true_l = []
	false_l = []
	for item in l:
		if f(item):
			true_l.append(item)
		else:
			false_l.append(item)
	return true_l, false_l


def try_or_else(getter, default):
	try:
		return getter()
	except:
		return default
