import serial, time
import serial.tools.list_ports

def FindMega():
    ports = list(serial.tools.list_ports.comports())

    for p in ports:
        if "0042" in p[2]:
            return (p[0])
        else:
            return ('NULL')
        
def FindUno():
    port = list(serial.tools.list_ports.comports())

    for p in port:
        if "0043" in p[2]:
            return (p[0])
        else:
            return ('NULL')
        
mega_port = FindMega()
time.sleep(4)
# uno_port = FindUno()

print("Mega Port: ", mega_port)#," Uno Port: ", uno_port)