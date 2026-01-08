`timescale 1ns / 1ps

module tb_spi_slave;

    // --- 1. Señales ---
    reg rst;
    reg sclk;
    reg ss;
    reg mosi;
    reg [7:0] data_in;

    wire miso;
    wire [7:0] data_out;
    wire [6:0] addr_out;
    wire write_enable;

    // Variable para guardar lo que lee el Maestro (MISO)
    reg [15:0] master_rx; 

    // --- 2. Instancia del Módulo ---
    // AVISO: Asegurate de usar el nombre correcto de tu modulo (spi_slave o spi_slave_FINAL)
    spi_slave u_spi_slave (
        .rst(rst),
        .sclk(sclk),
        .ss(ss),
        .mosi(mosi),
        .miso(miso),
        .data_in(data_in),
        .data_out(data_out),
        .addr_out(addr_out),
        .write_enable(write_enable)
    );

    // --- 3. Generación de Reloj ---
    initial sclk = 0;
    always #10 sclk = ~sclk;

    // --- 4. Tarea para Enviar y RECIBIR (Full Duplex) ---
    task send_frame(input [15:0] trama);
        integer i;
        begin
            @(negedge sclk); // Sincronización inicial
            ss = 0;
            
            // Limpiamos el registro de recepción del maestro
            master_rx = 16'h0000;

            for (i = 15; i >= 0; i = i - 1) begin
                // A) Flanco de BAJADA: Maestro pone dato en MOSI
                mosi = trama[i]; 
                
                // B) Flanco de SUBIDA: Maestro LEE dato de MISO
                @(posedge sclk); 
                
                // --- AQUI ESTA LA MAGIA ---
                // Leemos el cable MISO y lo guardamos.
                // Si es 'Z' (alta impedancia), lo tomamos como 0 para no tener errores de 'X'
                if (miso === 1'bz) 
                    master_rx = {master_rx[14:0], 1'b0};
                else
                    master_rx = {master_rx[14:0], miso};
                // --------------------------

                // C) Esperar ciclo completo
                @(negedge sclk); 
            end

            ss = 1;
            mosi = 0;
            #20;
        end
    endtask

    // --- 5. Bloque Principal ---
    initial begin
        $display("--- INICIANDO SIMULACION ---");
        
        // Inicialización
        ss = 1; mosi = 0; rst = 1; data_in = 8'h00;
        #100; rst = 0; #100;

        // ---------------------------------------------------------
        // TEST 1: ESCRITURA 
        // ---------------------------------------------------------
        $display("\nTest 1: Escribiendo 0xA3 en Direccion 0x05...");
        send_frame(16'h05A3);

        #1; 
        if (addr_out === 7'h05 && data_out === 8'hA3) 
            $display("  -> [EXITO] Escritura Correcta.");
        else 
            $display("  -> [FALLO] Escritura. Recibido: Addr=%h Data=%h", addr_out, data_out);

        // ---------------------------------------------------------
        // TEST 2: LECTURA
        // ---------------------------------------------------------
        data_in = 8'h55; // El sistema prepara el dato 55 para enviarnos
        
        $display("\nTest 2: Leyendo Direccion 0x05 (Esperamos 0x55 en MISO)...");
        // Mandamos comando de lectura (Bit 15 en 1)
        send_frame(16'h8500);
        
        // Verificamos qué recibió nuestra variable 'master_rx'
        // Nos importan los últimos 8 bits (el byte de datos)
        if (master_rx[7:0] === 8'h55) begin
            $display("  -> [EXITO] Lectura Correcta! MISO entrego 0x55.");
        end else begin
            $display("  -> [FALLO] Lectura Incorrecta.");
            $display("     Esperaba: 0x55");
            $display("     Recibio MISO: 0x%h", master_rx[7:0]);
            $display("     (Nota: Si recibiste AA o 2A, revisa el cambio de >7 a >8)");
        end

        #100;
        $display("\n--- FIN DE SIMULACION ---");
        $stop;
    end

endmodule