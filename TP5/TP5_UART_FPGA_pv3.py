import time
import serial
import sys

portUSB = sys.argv[1]

ser = serial.Serial(
    port='/dev/ttyUSB{}'.format(int(portUSB)),
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)

ser.isOpen()
ser.timeout = None
print("Puerto abierto y listo.")

colores = {
    0: 'apagado',
    1: 'rojo',
    2: 'verde',
    3: 'amarillo',
    4: 'azul',
    5: 'magenta',
    6: 'cyan',
    7: 'blanco'
}

print("\nComandos válidos:")
print("  L - Controlar un LED")
print("  S - Leer switches")
print("  exit - Salir del programa\n")

while True:
    cmd = input("<< ").strip().lower()

    if cmd == 'exit':
        ser.close()
        exit()

    elif cmd == 'l':
        try:
            led_id = int(input("ID de LED (0-3): "))
            if not 0 <= led_id <= 3:
                print("\u26a0\ufe0f LED fuera de rango.")
                continue

            print("Colores disponibles:")
            for valor, nombre in colores.items():
                print(f"  {valor} -> {nombre}")
            color_val = int(input("Ingrese color (por número): ").strip())

            if color_val not in colores:
                print("\u26a0\ufe0f Número de color no válido.")
                continue

            frame = bytes([led_id, color_val, 1])
            ser.write(frame)
            print(f"\u2705 Enviado: LED {led_id} → {colores[color_val].upper()}")

        except Exception as e:
            print("\u274c Entrada inválida. Intente de nuevo.")

    elif cmd == 's':
        frame = bytes([0, 0, 2])
        ser.write(frame)
        print("\u231b Esperando estado de switches...")

        ser.timeout = 1
        readData = ser.read(2)
        ser.timeout = None

        if len(readData) == 2:
            trama_rx = int.from_bytes(readData, 'big')
            device = (trama_rx >> 12) & 0x03
            data   = trama_rx & 0x0FFF

            sw_states = [(data >> i) & 1 for i in range(4)]
            estado_str = ", ".join([f"SW{i}: {'ON' if sw else 'OFF'}" for i, sw in enumerate(sw_states)])

            print(f"\U0001f7e2 Trama recibida: 0x{trama_rx:04X}")
            print(f"    Periférico : {device} (esperado: 1)")
            print(f"    Switches   : {data} (binario: {data:04b})")
            print(f"    Estado     : [{estado_str}]")
        else:
            print("\u274c No se recibieron los 2 bytes esperados")

    else:
        print("\u26a0\ufe0f Comando no reconocido.")
