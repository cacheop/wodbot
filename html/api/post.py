import requests
res = requests.post('http://localhost:5000/api', 
		json={"gen_length":200, "prime_word": "shots"})
		
if res.ok:
    print res.json()

