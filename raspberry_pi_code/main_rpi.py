from time import sleep
from machine import Pin

led = Pin("LED", Pin.OUT)
led.on()
sleep(1)
led.off()
sleep(4)
import web_server
