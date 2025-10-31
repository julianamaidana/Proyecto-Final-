#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_cache.h"
#include "xgpio.h"
#include "platform.h"
#include "xuartlite.h"
//#include "microblaze_sleep.h"

#define PORT_IN	 		XPAR_AXI_GPIO_0_BASEADDR //XPAR_GPIO_0_DEVICE_ID
#define PORT_OUT 		XPAR_AXI_GPIO_0_BASEADDR //XPAR_GPIO_0_DEVICE_ID

//Device_ID Operaciones
//#define def_SOFT_RST            0
//#define def_ENABLE_MODULES      1
//#define def_LOG_RUN             2
//#define def_LOG_READ            3

XGpio GpioOutput;
XGpio GpioParameter;
XGpio GpioInput;
u32 GPO_Value;
u32 GPO_Param;
XUartLite uart_module;

//Funcion para recibir 1 byte bloqueante
//XUartLite_RecvByte((&uart_module)->RegBaseAddress)

int main()
{
	init_platform();
	int Status;
	XUartLite_Initialize(&uart_module, 0);

	GPO_Value=0x00000000;
	GPO_Param=0x00000000;
	unsigned char cabecera[4];

	Status=XGpio_Initialize(&GpioInput, PORT_IN);
	if(Status!=XST_SUCCESS){
        return XST_FAILURE;
    }
	Status=XGpio_Initialize(&GpioOutput, PORT_OUT);
	if(Status!=XST_SUCCESS){
		return XST_FAILURE;
	}
	XGpio_SetDataDirection(&GpioOutput, 1, 0x00000000);
	XGpio_SetDataDirection(&GpioInput, 1, 0xFFFFFFFF);

	u32 value;
  unsigned char datos;
	while(1){
        //XUartLite_Recv(&uart_module, &(cabecera[0]), 1);
        //read(stdin,&cabecera[0],1);

    /* unsigned char datos_rec[3]; */
    /* short int i; */
    /* for(i=0;i<3;i++){ */
    /*   datos_rec[i]=XUartLite_RecvByte((&uart_module)->RegBaseAddress); */
    /* } */
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// ACA es donde se escribe toda la funcionalidad
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // CODIGO VIEJO POLA
		   //switch(cabecera[0]){
           //case '0':
           //    XGpio_DiscreteWrite(&GpioOutput,1, (u32) 0x00000249);
           //    break;
           //case '1':
           //    XGpio_DiscreteWrite(&GpioOutput,1, (u32) 0x00000492);
           //    break;
           //case '2':
           //    XGpio_DiscreteWrite(&GpioOutput,1, (u32) 0x00000924);
           //    break;
           //case '3':
           //    XGpio_DiscreteWrite(&GpioOutput,1, (u32) 0x00000000);
           //    value = XGpio_DiscreteRead(&GpioInput, 1);
           //    datos=(char)(value&(0x0000000F));
           //    while(XUartLite_IsSending(&uart_module)){}
           //    XUartLite_Send(&uart_module, &(datos),1);
           //    break;
		   //}
            unsigned char frame[3];
            for (int i = 0; i < 3; i++)
                frame[i] = XUartLite_RecvByte(uart_module.RegBaseAddress);

            unsigned char led_id = frame[0];
            unsigned char color  = frame[1];
            unsigned char cmd    = frame[2];

            if (cmd == 1) {
                if (led_id < 4) {
                    u32 mask = 0x7 << (led_id * 3);         // 3 bits por LED
                    GPO_Value &= ~mask;                     // apaga bits anteriores
                    GPO_Value |= ((color & 0x7) << (led_id * 3)); // enciende nuevos bits
                    XGpio_DiscreteWrite(&GpioOutput, 1, GPO_Value);
                }
            }
            else if (cmd == 2) {
                u32 value = XGpio_DiscreteRead(&GpioInput, 1);
                u16 trama = (1 << 12) | (value & 0x0FFF); // codifica respuesta: device=1
                unsigned char resp[2];
                resp[0] = (trama >> 8) & 0xFF;
                resp[1] = trama & 0xFF;

                while (XUartLite_IsSending(&uart_module)) {}
                XUartLite_Send(&uart_module, resp, 2);
            }
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// FIN de toda la funcionalidad
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        }
	
	cleanup_platform();
	return 0;
}
