`timescale 1ns / 1ps

module spi_slave (
    // --- Puertos del Sistema ---
    input  wire        rst,          // Reset global (Activo Alto)
    
    // --- Interfaz SPI (Física) ---
    input  wire        sclk,         // Reloj SPI (Desde el Maestro)
    input  wire        ss,           // Slave Select (Activo Bajo: 0 = seleccionado)
    input  wire        mosi,         // Entrada de datos serie
    output wire        miso,         // Salida de datos serie

    // --- Interfaz Interna (Hacia FSE-LMS) ---
    input  wire [7:0]  data_in,      // Dato que el sistema quiere enviar al maestro (Lectura)
    output reg  [7:0]  data_out,     // Dato recibido para guardar (Escritura)
    output reg  [6:0]  addr_out,     // Dirección del registro accedido
    output reg         write_enable  // Pulso: "1" cuando data_out es válido para escribir
);

    // --- Parámetros de la Trama (16 bits) ---
    // [15] = R/W | [14:8] = Address | [7:0] = Data
    localparam integer FRAME_SIZE = 16;
    localparam integer DATA_WIDTH = 8;
    localparam integer ADDR_WIDTH = 7;
    
    // Posiciones clave
    localparam BIT_RW   = 15;        // Posición del bit de control
    localparam CMD_READ = 1'b1;      // Valor 1 indica LECTURA
    
    // --- Registros Internos ---
    reg [FRAME_SIZE-1:0] shift_reg;  // Registro de desplazamiento principal (RX), lo que se manda por mosi
    reg [DATA_WIDTH-1:0] tx_reg;     // Buffer para transmitir datos (TX) miso
    reg [3:0]            bit_cnt;    // Contador de bits (0 a 15)
    
    // --- Lógica de Recepción (RX) - Flanco de SUBIDA --- escribiendo 
    // En SPI Modo 0, el maestro pone datos antes y el esclavo los lee en la subida .
    always @(posedge sclk or posedge rst) begin
        if (rst) begin
            shift_reg    <= 16'd0;
            bit_cnt      <= 4'd0;
            data_out     <= 8'd0;
            addr_out     <= 7'd0;
            write_enable <= 1'b0;
        end 
        else begin
            // Si SS está en alto , el esclavo está inactivo (Reset sincrónico)
            if (ss == 1'b1) begin
                bit_cnt      <= 4'd0;
                write_enable <= 1'b0;
                // shift_reg no necesita limpiarse, se sobrescribirá
            end 
            else begin
                // --- ESCLAVO SELECCIONADO (SS = 0) ---
                
                // 1. Desplazamiento (Shift): Entra MOSI por la derecha (LSB)
                // Usamos concatenación para meter el bit nuevo
                shift_reg <= {shift_reg[FRAME_SIZE-2:0], mosi};
                
                // 2. Contador
                bit_cnt <= bit_cnt + 1;
                
                // 3. Detección de FINAL DE TRAMA (Bit 15)
                // Como bit_cnt cuenta 0..15, cuando vale 15 terminamos el paquete.
                if (bit_cnt == 4'd15) begin
                    
                    // a) Extraer Dirección (Bits 14 al 8)
                    // Nota: Usamos shift_reg desplazado porque el MSB original ya "viajó" a la izq.
                    // En el ciclo 15, el bit 14 (ADDR MSB) está en shift_reg[14].
                    addr_out <= shift_reg[13:7]; 
                    
                    // b) Extraer Dato (Bits 7 al 0)
                    // El último bit (mosi) es el bit 0 del dato.
                    data_out <= {shift_reg[6:0], mosi};
                    
                    // c) Lógica de Escritura
                    // El bit R/W fue el primero en entrar (hace 15 ciclos).
                    // Si ese bit era '0' (Write), activamos el enable.
                    // El bit R/W está actualmente en la posición tope del shift histórico.
                    if (shift_reg[14] != CMD_READ) begin
                        write_enable <= 1'b1;
                    end
                end 
                else begin
                    write_enable <= 1'b0; // El pulso dura solo 1 ciclo
                end
            end
        end
    end

    // --- Lógica de Transmisión (TX) - Flanco de BAJADA ---
    // En SPI Modo 0, el esclavo debe cambiar MISO en la bajada para que 
    // el maestro lo lea estable en la siguiente subida.
    always @(negedge sclk or posedge rst) begin
        if (rst) begin
            tx_reg <= 8'd0;
        end
        else begin
            if (ss == 1'b0) begin
                // Estamos activos.
                
                // PUNTO DE CRUCE: Fin de la Cabecera (Bit 7 / Address LSB)
                // Aquí decidimos qué cargar para enviar en la segunda mitad.
                if (bit_cnt == 4'd7) begin // Justo terminamos de recibir Dirección
                    
                    // Miramos el bit de R/W que recibimos al principio.
                    // En este punto, el bit R/W está en shift_reg[6] (se desplazó 7 veces).
                    if (shift_reg[6] == CMD_READ) begin
                        tx_reg <= data_in; // ¡Cargamos el dato del sistema!
                    end else begin
                        tx_reg <= 8'd0;    // Si es escritura, mandamos 0
                    end
                end
                else if (bit_cnt > 8) begin
                    // Durante la fase de datos, desplazamos hacia afuera
                    tx_reg <= {tx_reg[6:0], 1'b0};
                end
            end
        end
    end

    // --- Asignación de Salida MISO ---
    // MISO saca el bit más significativo (MSB) del registro de transmisión
    // Solo manejamos la línea cuando SS está bajo (Tri-state si no).
    assign miso = (ss == 1'b0 && bit_cnt > 7) ? tx_reg[7] : 1'bz;

endmodule