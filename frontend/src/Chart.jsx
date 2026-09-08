import Plot from "react-plotly.js";

export default function Chart({ figure }) {
  if (!figure) return null;

  return (
    <div className="chart-frame scale-in">
      <Plot
        data={figure.data}
        layout={{ ...figure.layout, autosize: true }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
