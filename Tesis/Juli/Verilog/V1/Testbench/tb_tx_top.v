`timescale 1ns/1ps

module tb_tx_top_check;

    // --- 1. Señales ---
    reg clk;
    reg reset;
    wire signed [15:0] sI_out;
    wire signed [15:0] sQ_out;

    // Variables para lectura de archivo y control
    integer file_id;
    integer scan_res;
    integer i;
    integer errors;
    
    // Variables temporales para almacenar el valor esperado del archivo
    reg [15:0] expected_I_hex;
    reg [15:0] expected_Q_hex;
    wire signed [15:0] expected_I_signed;
    wire signed [15:0] expected_Q_signed;

    // Asignación para ver los valores esperados como decimales con signo en la onda
    assign expected_I_signed = expected_I_hex;
    assign expected_Q_signed = expected_Q_hex;

    // --- 2. Instancia del DUT (Tu diseño) ---
    tx_top dut (
        .clk(clk),
        .reset(reset),
        .sI_out(sI_out),
        .sQ_out(sQ_out)
    );

    // --- 3. Generador de Reloj (100 MHz) ---
    always #5 clk = ~clk;

    // --- 4. Proceso de Prueba ---
    initial begin
        // Inicialización
        clk = 0;
        reset = 1;
        errors = 0;

        // Abrir el archivo de referencia
        // NOTA: Si Vivado no lo encuentra, usa la ruta absoluta (ej: "C:/proyectos/verilog_ref.mem")
        file_id = $fopen("verilog_ref.mem", "r");
        
        if (file_id == 0) begin
            $display("ERROR FATAL: No se pudo abrir 'verilog_ref.mem'.");
            $display("Asegurate de correr el script Python y poner el archivo en la carpeta correcta.");
            $finish;
        end

        $display("---------------------------------------");
        $display("--- INICIO DE VALIDACION AUTOMATICA ---");
        $display("---------------------------------------");

        // Secuencia de Reset
        #100;
        reset = 0;

        // Sincronización: El PRBS tarda 1 ciclo en sacar el primer dato válido tras el reset
        @(posedge clk);
        scan_res = $fscanf(file_id, "%h %h\n", expected_I_hex, expected_Q_hex);
        // Bucle de comparación
        // Leemos hasta el final del archivo (EOF)
        while (!$feof(file_id)) begin
            
            // 1. Leer una línea del archivo (formato hex hex)
            scan_res = $fscanf(file_id, "%h %h\n", expected_I_hex, expected_Q_hex);
            
            // Si la lectura fue exitosa (2 items leídos)
            if (scan_res == 2) begin
                
                // 2. Esperar a que el DUT estabilice la salida (flanco de bajada)
                @(negedge clk);
                
                // 3. Comparar salida real vs esperada
                if ((sI_out !== expected_I_signed) || (sQ_out !== expected_Q_signed)) begin
                    $display("[ERROR] Tiempo: %0t | I: Real=%d Esp=%d | Q: Real=%d Esp=%d", 
                             $time, sI_out, expected_I_signed, sQ_out, expected_Q_signed);
                    errors = errors + 1;
                end
                
                // Volver al flanco de subida para el siguiente ciclo
                @(posedge clk);
            end
        end

        // --- Reporte Final ---
        $fclose(file_id);
        
        $display("\n---------------------------------------");
        if (errors == 0) begin
            $display(" RESULTADO: PASS (Exito Total)");
            $display(" Todos los vectores coincidieron con Python.");
        end else begin
            $display(" RESULTADO: FAIL");
            $display(" Se encontraron %0d errores.", errors);
        end
        $display("---------------------------------------\n");
        
        $stop;
    end

endmodule