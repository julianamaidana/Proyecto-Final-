#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_cache.h"
#include "xgpio.h"
#include "platform.h"
#include "xuartlite.h"
#include "microblaze_sleep.h"

// Ambas apuntan al mismo periférico
#define GPIO_PORT       XPAR_AXI_GPIO_0_BASEADDR 

// MÁSCARA DE DIRECCIÓN:
// Basado en el Verilog (fpga_procom.v):
// gpi0[3:0] -> bits 0-3 son ENTRADA (1)
// gpo0[..._] -> bits 4-31 son SALIDA (0)
#define GPIO_DIRECTION_MASK 0x0000000F


// --- Definiciones de LEDs (esto estaba bien) ---
#define COLOR_OFF   (0x0) // 000
#define COLOR_RED   (0x1) // 001
#define COLOR_GREEN (0x2) // 010
#define COLOR_BLUE  (0x4) // 100
#define COLOR_WHITE (0x7) // 111

#define LED0_SHIFT 0
#define LED1_SHIFT 3
#define LED2_SHIFT 6
#define LED3_SHIFT 9

#define LED0_MASK (~(0x7 << LED0_SHIFT))
#define LED1_MASK (~(0x7 << LED1_SHIFT))
#define LED2_MASK (~(0x7 << LED2_SHIFT))
#define LED3_MASK (~(0x7 << LED3_SHIFT))
// --- Fin Definiciones ---

// --- CORRECCIÓN 1: Usar UNA sola instancia de GPIO ---
XGpio Gpio; 
u32 GPO_Value; // Estado de los LEDs
XUartLite uart_module;

int main()
{
    init_platform();
    int Status;
    XUartLite_Initialize(&uart_module, 0);

    GPO_Value = 0x00000000; // Todos los LEDs apagados
    unsigned char cabecera[2]; // Nuestro protocolo de 2 bytes [LED][COLOR]

    // --- CORRECCIÓN 2: Inicializar UNA sola instancia ---
    Status = XGpio_Initialize(&Gpio, GPIO_PORT);
    if(Status != XST_SUCCESS){
        return XST_FAILURE;
    }
    
    // --- CORRECCIÓN 3: Establecer la MÁSCARA de dirección ---
    // Solo usamos el Canal 1
    XGpio_SetDataDirection(&Gpio, 1, GPIO_DIRECTION_MASK);

    u32 value;
    unsigned char datos;

    // Escribir estado inicial (0)
    XGpio_DiscreteWrite(&Gpio, 1, GPO_Value);

    while(1){
    
    // Espera los 2 bytes de nuestro protocolo (ej: "0R")
    XUartLite_Recv( &uart_module, cabecera, (unsigned int) 2);
    
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// ACA es donde se escribe toda la funcionalidad
// (Esta lógica estaba bien, solo cambiamos GpioOutput por Gpio)
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        u32 color_bits = 0; 

        // 1. Convertir el Dato (cabecera[1]) a bits de color
        switch(cabecera[1]) {
            case 'R': color_bits = COLOR_RED;   break;
            case 'G': color_bits = COLOR_GREEN; break;
            case 'B': color_bits = COLOR_BLUE;  break;
            case 'W': color_bits = COLOR_WHITE; break;
            case 'O': color_bits = COLOR_OFF;   break;
            default:  color_bits = COLOR_OFF;   break;
        }

        // 2. Actuar segun el Comando (cabecera[0])
        switch(cabecera[0]){
            
            case '0': // Controlar LED 0
                GPO_Value = (GPO_Value & LED0_MASK) | (color_bits << LED0_SHIFT);
                XGpio_DiscreteWrite(&Gpio, 1, GPO_Value);
                break;
                
            case '1': // Controlar LED 1
                GPO_Value = (GPO_Value & LED1_MASK) | (color_bits << LED1_SHIFT);
                XGpio_DiscreteWrite(&Gpio, 1, GPO_Value);
                break;

            case '2': // Controlar LED 2
                GPO_Value = (GPO_Value & LED2_MASK) | (color_bits << LED2_SHIFT);
                XGpio_DiscreteWrite(&Gpio, 1, GPO_Value);
                break;

            case '3': // Controlar LED 3
                GPO_Value = (GPO_Value & LED3_MASK) | (color_bits << LED3_SHIFT);
                XGpio_DiscreteWrite(&Gpio, 1, GPO_Value);
                break;
                
            case 'c': // 'c'lear (limpiar) todos los LEDs
                GPO_Value = 0x00000000;
                XGpio_DiscreteWrite(&Gpio, 1, GPO_Value);
                break;

            case 'r': // 'r'ead (leer) inputs
                // --- CORRECCIÓN 4: Leer del Canal 1 ---
                value = XGpio_DiscreteRead(&Gpio, 1); 
                datos = (char)(value & 0x0000000F); // Leemos los 4 bits de entrada
                while(XUartLite_IsSending(&uart_module)){}
                XUartLite_Send(&uart_module, &datos, 1);
                break;
        }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// FIN de toda la funcionalidad
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    }
    
    cleanup_platform();
    return 0;
}