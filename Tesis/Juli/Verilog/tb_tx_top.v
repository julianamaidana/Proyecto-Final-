`timescale 1ns/1ps

module tb_tx_top;

    // 1. Declaración de señales (Inputs como reg, Outputs como wire)
    reg clk;
    reg reset;
    wire signed [15:0] sI_out;
    wire signed [15:0] sQ_out;

    // Variables para manejo de archivo
    integer f;
    integer i;

    // 2. Instanciación del DUT (Device Under Test)
    tx_top dut (
        .clk(clk),
        .reset(reset),
        .sI_out(sI_out),
        .sQ_out(sQ_out)
    );

    // 3. Generación de Reloj (100 MHz -> periodo 10ns)
    always #5 clk = ~clk;

    // 4. Proceso Principal
    initial begin
        // Inicialización
        clk = 0;
        reset = 1; // Reset activado al inicio
        f = $fopen("salida_verilog.txt", "w"); // Abrir archivo para escritura
        
        // Escribir cabecera en el archivo (opcional, para claridad)
        // $fwrite(f, "n,sI,sQ\n"); 

        $display("--- Iniciando Simulación ---");
        
        // Mantener reset por 100ns
      #100;
        reset = 0; // Soltar reset
        $display("--- Reset liberado ---");

        // Esperar un ciclo para que el PRBS arranque
        @(posedge clk);

        // 5. Bucle de captura de datos
        // Vamos a capturar 50 muestras para verificar visualmente
        for (i = 0; i < 50; i = i + 1) begin
            // Esperamos al flanco de subida
            @(posedge clk); 
            
            // Esperamos 1ns extra para leer el dato estable (post-clock)
            #1; 
            
            // Imprimir en consola de Vivado
            $display("Muestra %0d: I = %d, Q = %d", i, sI_out, sQ_out);
            
            // Guardar en archivo (Formato: sI, sQ)
            // Usamos %0d para decimal con signo
            $fwrite(f, "%0d,%0d\n", sI_out, sQ_out);
        end

        // Cerrar archivo y terminar
        $fclose(f);
        $display("--- Simulación Finalizada. Datos guardados en salida_verilog.txt ---");
        $stop;
    end

endmodule