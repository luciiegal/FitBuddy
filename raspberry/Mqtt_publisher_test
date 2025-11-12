import paho.mqtt.client as mqtt
import time 

BROKER = "83.30.29.68" #Ip de la VM
PORT = 1883
TOPIC = "raspi/test" 

try: 
	client = mqtt.Client()
	client.connect(BROKER, PORT,keepalive=60)
	print("Connect")
	
	while True:
		message = "Hello world"
		client.publish(TOPIC,message)
		print("Message envoye")
		time.sleep(2)
		
except KeyboardInterrupt:
	print("Arret")
	
except Exception as e:
	print("Erreur ",e)

finally :
	client.disconnect()
