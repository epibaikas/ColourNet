import network
import socket
from time import sleep
import machine
from machine import Pin
import json

from neopixel import Neopixel

# Set the SSID and password of your WiFi network
ssid = 'SSID'
password = 'password'

led = Pin("LED", Pin.OUT)

numpix = 54
strip = Neopixel(numpix, 0, 28, "RGB")
strip.brightness(255)

weight_leds = {'fc1_weight_0': (5, 36),
               'fc1_weight_1': (4, 28),
               'fc1_weight_2': (3, 20),
               'fc1_weight_3': (9, 35),
               'fc1_weight_4': (10, 27),
               'fc1_weight_5': (11, 19),
               'fc1_weight_6': (15, 34),
               'fc1_weight_7': (16, 26),
               'fc1_weight_8': (17, 18),
               'fc2_weight_0': (41, 47),
               'fc2_weight_1': (40, 53),
               'fc2_weight_2': (33, 46),
               'fc2_weight_3': (32, 52),
               'fc2_weight_4': (25, 45),
               'fc2_weight_5': (24, 51)
               }

node_leds = {'node_0': 1,  # (0, 1, 2),
             'node_1': 7,  # (6, 7, 8),
             'node_2': 13,  # (12, 13, 14),
             'node_3': 38,  # (37, 38, 39),
             'node_4': 30,  # (29, 30, 31),
             'node_5': 22,  # (21, 22, 23),
             'node_6': 43,  # (42, 43, 44),
             'node_7': 49  # (48, 49, 50)
             }

input_layer_leds = [node_leds['node_0'], node_leds['node_1'], node_leds['node_2']]
hidden_layer_leds = [node_leds['node_3'], node_leds['node_4'], node_leds['node_5']]
output_layer_leds = [node_leds['node_6'], node_leds['node_7']]

delay = 0.08


def set_colour_of_scattered_leds(led_idx, colour):
    for idx in led_idx:
        strip[idx] = colour
    strip.show()


def update_led_strip(weight_colours):
    # Turn the leds of all nodes on
    set_colour_of_scattered_leds(input_layer_leds, (255, 255, 255))
    set_colour_of_scattered_leds(hidden_layer_leds, (255, 255, 255))
    set_colour_of_scattered_leds(output_layer_leds, (255, 255, 255))
    sleep(delay)

    set_colour_of_scattered_leds(input_layer_leds, (0, 0, 0))
    sleep(delay)
    set_colour_of_scattered_leds(hidden_layer_leds, (0, 0, 0))
    sleep(delay)
    set_colour_of_scattered_leds(output_layer_leds, (0, 0, 0))
    sleep(delay)
    set_colour_of_scattered_leds(output_layer_leds, (255, 255, 255))
    sleep(delay)
    set_colour_of_scattered_leds(hidden_layer_leds, (255, 255, 255))
    sleep(delay)
    set_colour_of_scattered_leds(input_layer_leds, (255, 255, 255))
    sleep(delay)

    for i in range(9):
        weight_str = 'fc1_weight_' + str(i)
        leds = weight_leds[weight_str]
        for led in leds:
            colour_str = weight_colours[weight_str]
            rgb_values = colour_str.split(',')
            colour = tuple(int(value) for value in rgb_values)
            strip.set_pixel(led, colour)

    for i in range(6):
        weight_str = 'fc2_weight_' + str(i)
        leds = weight_leds[weight_str]
        for led in leds:
            colour_str = weight_colours[weight_str]
            rgb_values = colour_str.split(',')
            colour = tuple(int(value) for value in rgb_values)
            strip.set_pixel(led, colour)

    strip.show()


def connect():
    # Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    while wlan.isconnected() == False:
        led.on()
        print('Waiting for connection...')
        sleep(0.5)
        led.off()
        sleep(0.5)
    ip = wlan.ifconfig()[0]
    print(f'Connected on {ip}')
    return ip


def open_socket(ip):
    # Open a socket
    address = (ip, 80)
    connection = socket.socket()
    connection.bind(address)
    connection.listen(5)
    print(connection)
    return connection


def webpage():
    # Template HTML
    html = f"""
            <!DOCTYPE html>
            <html>
            <body>
            <p>ColourNet Webpage</p>
            </body>
            </html>   
            """
    return str(html)


def serve(connection):
    # Start a web server
    led.off()
    while True:
        client = connection.accept()[0]
        request = client.recv(1024)
        request = str(request)

        try:
            request_type = request.split()[1]
            print(request_type)
        except IndexError:
            pass

        if request_type == '/json':
            data_json = request.split()[6]
            weights_colours = json.loads(data_json)
            update_led_strip(weights_colours)

        html = webpage()
        client.send(html)
        client.close()


try:
    ip = connect()
    connection = open_socket(ip)
    serve(connection)
except KeyboardInterrupt:
    machine.reset()
