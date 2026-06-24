`timescale 1ns / 1ps
/**
 * =============================================================================
 * affinity_scorer_top.v - AXI-Stream Top-Level (No embedding_loader)
 * =============================================================================
 * DVCon India 2026 - Stage 2B
 * Task-Aware Object Detection - Affinity Scorer Accelerator
 *
 * Embedding buffer logic is inlined directly into the input parser FSM.
 * No separate embedding_loader module needed.
 *
 * Input Stream Protocol:
 *   TYPE 0x01 → task embedding  (EMB_DIM signed bytes follow)
 *   TYPE 0x02 → label embedding (EMB_DIM signed bytes follow)
 *   TYPE 0x03 → meta            (5 bytes: conf_hi, conf_lo,
 *                                          boost_hi, boost_lo, flags[0]=penalty)
 *   TYPE 0x04 → n_objects       (2 bytes: hi, lo)
 *   TYPE 0x05 → start           (no extra bytes, triggers FSM immediately)
 *
 * Output Stream:
 *   6 bytes when done: best_object[15:8], best_object[7:0],
 *                      best_score[31:24], best_score[23:16],
 *                      best_score[15:8],  best_score[7:0]
 *
 * Target : Genesys-2 (Xilinx Kintex-7 XC7K325T)
 * Tool   : Vivado 2025.1
 * =============================================================================
 */

module affinity_scorer_top #(
    parameter EMB_DIM = 8
)(
    input  wire        axi_clk,
    input  wire        axi_reset_n,

    // ─── Slave AXI-Stream (input) ────────────────────────────────────────
    input  wire        i_data_valid,
    input  wire [7:0]  i_data,
    output wire        o_data_ready,

    // ─── Master AXI-Stream (output) ──────────────────────────────────────
    output reg         o_data_valid,
    output reg  [7:0]  o_data,
    input  wire        i_data_ready,

    // ─── Interrupt ───────────────────────────────────────────────────────
    output reg         o_intr
);

    // ─── Internal Reset (active high for submodules) ──────────────────────
    wire rst = !axi_reset_n;

    // ─── Inlined Embedding Buffers ────────────────────────────────────────
    reg signed [7:0]        task_buf  [0:EMB_DIM-1];
    reg signed [7:0]        label_buf [0:EMB_DIM-1];
    reg                     task_full;
    reg                     label_full;

    // Flatten buffers to flat buses for MAC
    wire signed [8*EMB_DIM-1:0] task_emb_flat;
    wire signed [8*EMB_DIM-1:0] label_emb_flat;

    genvar gi;
    generate
        for (gi = 0; gi < EMB_DIM; gi = gi + 1) begin : FLATTEN
            assign task_emb_flat [8*gi +: 8] = task_buf[gi];
            assign label_emb_flat[8*gi +: 8] = label_buf[gi];
        end
    endgenerate

    // ─── Input Parser FSM ─────────────────────────────────────────────────
    localparam TYPE_IDLE      = 3'd0;
    localparam TYPE_TASK_EMB  = 3'd1;
    localparam TYPE_LABEL_EMB = 3'd2;
    localparam TYPE_META      = 3'd3;
    localparam TYPE_N_OBJECTS = 3'd4;
    localparam TYPE_START     = 3'd5;

    reg [2:0]              rx_state;
    reg [$clog2(EMB_DIM):0] rx_count;

    // Registers assembled from stream
    reg signed [15:0]      confidence_reg;
    reg signed [15:0]      boost_reg;
    reg                    penalty_reg;
    reg        [15:0]      n_objects_reg;
    reg                    start_reg;      // one-cycle pulse

    // Always ready to accept input
    assign o_data_ready = 1'b1;

    integer j;

    always @(posedge axi_clk) begin
        if (rst) begin
            rx_state       <= TYPE_IDLE;
            rx_count       <= 0;
            confidence_reg <= 16'sd0;
            boost_reg      <= 16'sd0;
            penalty_reg    <= 1'b0;
            n_objects_reg  <= 16'd0;
            start_reg      <= 1'b0;
            task_full      <= 1'b0;
            label_full     <= 1'b0;
            for (j = 0; j < EMB_DIM; j = j + 1) begin
                task_buf[j]  <= 8'sd0;
                label_buf[j] <= 8'sd0;
            end
        end
        else begin
            // Default: deassert one-cycle signals
            start_reg <= 1'b0;

            if (i_data_valid && o_data_ready) begin
                case (rx_state)

                    TYPE_IDLE: begin
                        rx_count  <= 0;
                        
                        case (i_data)
                            8'h01: rx_state <= TYPE_TASK_EMB;
                            8'h02: rx_state <= TYPE_LABEL_EMB;
                            8'h03: rx_state <= TYPE_META;
                            8'h04: rx_state <= TYPE_N_OBJECTS;
                            8'h05: begin
                                start_reg <= 1'b1;
                                rx_state  <= TYPE_IDLE;
                            end
                            default: rx_state <= TYPE_IDLE;
                        endcase
                    end

                    TYPE_TASK_EMB: begin
                        // Write directly into task_buf at rx_count index
                        task_buf[rx_count] <= $signed(i_data);
                        rx_count           <= rx_count + 1;
                        if (rx_count == EMB_DIM - 1) begin
                            task_full <= 1'b1;
                            rx_state  <= TYPE_IDLE;
                        end
                    end

                    TYPE_LABEL_EMB: begin
                        // Write directly into label_buf at rx_count index
                        label_buf[rx_count] <= $signed(i_data);
                        rx_count            <= rx_count + 1;
                        if (rx_count == EMB_DIM - 1) begin
                            label_full <= 1'b1;
                            rx_state   <= TYPE_IDLE;
                        end
                    end

                    TYPE_META: begin
                        rx_count <= rx_count + 1;
                        case (rx_count)
                            0: confidence_reg[15:8] <= $signed(i_data);
                            1: confidence_reg[7:0]  <= i_data;
                            2: boost_reg[15:8]      <= $signed(i_data);
                            3: boost_reg[7:0]       <= i_data;
                            4: begin
                                penalty_reg <= i_data[0];
                                rx_state    <= TYPE_IDLE;
                            end
                        endcase
                    end

                    TYPE_N_OBJECTS: begin
                        rx_count <= rx_count + 1;
                        case (rx_count)
                            0: n_objects_reg[15:8] <= i_data;
                            1: begin
                                n_objects_reg[7:0] <= i_data;
                                rx_state           <= TYPE_IDLE;
                            end
                        endcase
                    end

                    default: rx_state <= TYPE_IDLE;

                endcase
            end
        end
    end

    // ─── Core Compute Submodules ──────────────────────────────────────────
    wire        mac_en, score_en, max_en;
    wire        mac_done, score_done, max_done;
    wire        done;
    wire [15:0] cur_obj_idx;

    wire signed [31:0] dot_product;
    wire signed [31:0] final_score;
    wire signed [31:0] best_score;
    wire        [15:0] best_object;

    wire emd_signal;
    assign emd_signal = task_full & label_full;

    control FSM (
        .clk      (axi_clk),
        .reset    (!axi_reset_n),
        .start    (start_reg),
        .n_objects(n_objects_reg),
        .emd      (emd_signal),
        .mac_done (mac_done),
        .score_done(score_done),
        .max_done (max_done),
        .mac_en   (mac_en),
        .score_en (score_en),
        .max_en   (max_en),
        .obj_out  (cur_obj_idx),
        .done     (done)
    );

    MAC mac (
        .clk           (axi_clk),
        .reset         (!axi_reset_n),
        .task_emb_flat (task_emb_flat),
        .label_emb_flat(label_emb_flat),
        .i_data_valid  (mac_en),
        .dot_product   (dot_product),
        .o_data_valid  (mac_done)
    );

    score Score (
        .clk         (axi_clk),
        .reset       (!axi_reset_n),
        .dot         (dot_product),
        .confidence  (confidence_reg),
        .boost       (boost_reg),
        .penalty     (penalty_reg),
        .i_data_valid(mac_done),
        .final_score (final_score),
        .o_data_valid(score_done)
    );

    max_score MS (
        .clk         (axi_clk),
        .reset       (!axi_reset_n),
        .i_data_valid(score_done),
        .final_score (final_score),
        .object_id   (cur_obj_idx),
        .o_data_valid(max_done),
        .best_score  (best_score),
        .best_object (best_object)
    );

    // ─── Output Stream ────────────────────────────────────────────────────
    // 6-byte packet: best_object[15:8], best_object[7:0],
    //                best_score[31:24], best_score[23:16],
    //                best_score[15:8],  best_score[7:0]

    localparam OUT_IDLE = 2'd0;
    localparam OUT_SEND = 2'd1;

    reg [1:0]  tx_state;
    reg [2:0]  tx_count;
    reg [47:0] tx_buf;

    always @(posedge axi_clk) begin
        if (rst) begin
            tx_state     <= OUT_IDLE;
            tx_count     <= 3'd0;
            tx_buf       <= 48'd0;
            o_data_valid <= 1'b0;
            o_data       <= 8'd0;
            o_intr       <= 1'b0;
        end
        else begin
            o_intr <= 1'b0;

            case (tx_state)

                OUT_IDLE: begin
                    o_data_valid <= 1'b0;
                    if (done) begin
                        tx_buf   <= {best_object, best_score};
                        tx_count <= 3'd0;
                        tx_state <= OUT_SEND;
                        o_intr   <= 1'b1;
                    end
                end

                OUT_SEND: begin
                    if (i_data_ready) begin
                        o_data_valid <= 1'b1;
                        o_data       <= tx_buf[47:40];
                        tx_buf       <= {tx_buf[39:0], 8'd0};
                        tx_count     <= tx_count + 1'b1;
                        if (tx_count == 3'd5)
                            tx_state <= OUT_IDLE;
                    end
                    else begin
                        o_data_valid <= 1'b0;
                    end
                end

                default: tx_state <= OUT_IDLE;

            endcase
        end
    end

endmodule
