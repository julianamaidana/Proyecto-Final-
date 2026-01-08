`timescale 1ns / 1ps

module tb_os_circular_buffer;

    // --- 1. Declaración de señales ---
    reg clk;
    reg rst;
    reg valid_in;
    reg [15:0] data_in; // Asumo WN=16

    wire valid_out_fft;
    wire [15:0] data_out_fft;
    wire start_fft;

    // --- 2. Instancia del Módulo a Probar (DUT) ---
    os_circular_buffer #(
        .WN(16)
    ) dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .data_in(data_in),
        .valid_out_fft(valid_out_fft),
        .data_out_fft(data_out_fft),
        .start_fft(start_fft)
    );

    // --- 3. Generación de Reloj (100 MHz - 10ns periodo) ---
    always #5 clk = ~clk;

    // --- 4. Estímulos de Prueba ---
    integer i;
    
    initial begin
        // Inicialización
        clk = 0;
        rst = 1;
        valid_in = 0;
        data_in = 0;

        // Soltamos el Reset
        #100;
        rst = 0;
        #20;

        // --- PRUEBA 1: Primer Bloque (Datos 1 a 16) ---
        $display("--- Iniciando Bloque 1 (Entrada 1..16) ---");
        
        for (i = 1; i <= 16; i = i + 1) begin
            @(posedge clk);
            valid_in = 1;
            data_in = i; // Enviamos 1, 2, 3...
        end
        
        // Cortamos entrada para dejarlo procesar
        @(posedge clk);
        valid_in = 0;
        data_in = 0;

        // Esperamos a que termine de enviar a la FFT (aprox 40 ciclos)
        wait(valid_out_fft == 1); // Empieza a salir
        wait(valid_out_fft == 0); // Termina de salir
        #50; // Un respiro

        // --- PRUEBA 2: Segundo Bloque (Datos 17 a 32) ---
        $display("--- Iniciando Bloque 2 (Entrada 17..32) ---");

        for (i = 17; i <= 32; i = i + 1) begin
            @(posedge clk);
            valid_in = 1;
            data_in = i; // Enviamos 17, 18, 19...
        end

        @(posedge clk);
        valid_in = 0;
        
        wait(valid_out_fft == 0); // Esperar fin de envío
        #50;

        // --- PRUEBA 3: Tercer Bloque (Datos 33 a 48) ---
        // Acá comprobamos que el puntero circular dio la vuelta correctamente
        $display("--- Iniciando Bloque 3 (Entrada 33..48) ---");

        for (i = 33; i <= 48; i = i + 1) begin
            @(posedge clk);
            valid_in = 1;
            data_in = i;
        end
        
        @(posedge clk);
        valid_in = 0;

        #500;
        $finish;
    end

    // --- 5. Monitor Visual (Opcional si no usas Waveform) ---
    always @(posedge clk) begin
        if (valid_out_fft) begin
            $display("Time %t | Salida FFT: %d", $time, data_out_fft);
        end
        if (start_fft) begin
            $display("Time %t | --- START FFT PULSE ---", $time);
        end
    end

endmodule