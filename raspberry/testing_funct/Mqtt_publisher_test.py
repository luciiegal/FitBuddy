import paho.mqtt.client as mqtt
import time 

BROKER = "20.251.170.166"
PORT = 1883
TOPIC = "test" 

try: 
	client = mqtt.Client()
	client.connect(BROKER, PORT,keepalive=60)
	print("Connect")
	
	while True:
		message = "Rep : 10, Force : 50 N, Power : 50 W"
		client.publish(TOPIC,message)
		print("Message envoye")
		time.sleep(2)
		
except KeyboardInterrupt:
	print("Arret")
	
except Exception as e:
	print("Erreur ",e)

finally :
	client.disconnect()
