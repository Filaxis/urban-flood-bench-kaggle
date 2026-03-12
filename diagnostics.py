import hashlib
print(hashlib.md5(open("src/ufb/infer/rollout.py","rb").read()).hexdigest())