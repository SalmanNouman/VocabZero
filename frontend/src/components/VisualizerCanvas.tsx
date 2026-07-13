import React, { useRef, useEffect } from "react";

interface VisualizerCanvasProps {
  analyserNode: AnalyserNode | null;
  isRecording: boolean;
}

export const VisualizerCanvas: React.FC<VisualizerCanvasProps> = ({
  analyserNode,
  isRecording,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);

      const currentW = canvas.width / window.devicePixelRatio;
      const currentH = canvas.height / window.devicePixelRatio;

      if (!isRecording || !analyserNode) {
        ctx.fillStyle = "#f8fafd";
        ctx.fillRect(0, 0, currentW, currentH);
        ctx.strokeStyle = "rgba(113, 128, 154, 0.28)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, currentH / 2);
        ctx.lineTo(currentW, currentH / 2);
        ctx.stroke();
        return;
      }

      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserNode.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "rgba(248, 250, 253, 0.4)";
      ctx.fillRect(0, 0, currentW, currentH);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#2563eb";
      ctx.shadowBlur = 8;
      ctx.shadowColor = "rgba(37, 99, 235, 0.28)";

      ctx.beginPath();
      const sliceWidth = currentW / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * currentH) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(currentW, currentH / 2);
      ctx.stroke();
    };

    draw();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [analyserNode, isRecording]);

  return (
    <div className="relative w-full h-32 rounded-lg overflow-hidden border border-border bg-surface-sunken">
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
};
