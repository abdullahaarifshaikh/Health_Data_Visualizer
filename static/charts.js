/* Charts accept calculated data, never executable model-generated options. */
window.HealthLensCharts = {
  render(container, spec) {
    if (!window.echarts) {
      container.textContent = 'The interactive chart library could not load. Refresh the page to retry.';
      return;
    }
    const chart = echarts.init(container, null, {renderer: 'canvas'});
    const color = ['#176b59', '#78ab95', '#a6c6a7', '#c4d28d', '#d6b76e', '#94b6c5', '#b6a3c0', '#d19c89'];
    const option = {
      color, animationDuration: 350,
      title: {text: spec.title || '', left: 12, top: 12, textStyle: {fontSize: 14, color: '#203e3a', fontWeight: 600},
              subtext: spec.displayed_points < spec.total_points ? `${spec.displayed_points} points displayed; calculations use all ${spec.total_points} complete pairs` : ''},
      tooltip: {trigger: spec.type === 'scatter' || spec.type === 'pie' ? 'item' : 'axis', renderMode: 'richText'},
      grid: {left: 75, right: 30, top: 80, bottom: 80, containLabel: true},
      toolbox: {right: 10, top: 36, feature: {restore: {}, saveAsImage: {pixelRatio: 2, title: 'Save chart'}}},
      aria: {enabled: true, description: spec.title}
    };
    if (spec.type === 'heatmap') {
      option.tooltip = {position: 'top', renderMode: 'richText'};
      option.xAxis = {type: 'category', data: spec.labels, axisLabel: {rotate: 25, fontSize: 10, width: 90, overflow: 'truncate'}};
      option.yAxis = {type: 'category', data: spec.labels, axisLabel: {fontSize: 10, width: 100, overflow: 'truncate'}};
      option.grid = {left: 115, right: 30, top: 65, bottom: 100};
      option.visualMap = {min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 5,
        inRange: {color: ['#a85656', '#f5f4eb', '#176b59']}};
      option.series = [{type: 'heatmap', data: spec.points, label: {show: spec.labels.length <= 6,
        formatter: p => Number(p.value[2]).toFixed(2)}, emphasis: {itemStyle: {borderColor: '#203e3a', borderWidth: 1}}}];
    } else if (spec.type === 'pie') {
      option.series = [{type: 'pie', radius: ['28%', '60%'], center: ['50%', '55%'],
        data: (spec.labels || []).map((name, i) => ({name, value: spec.values[i]})),
        label: {formatter: '{b}\n{d}%', fontSize: 11}, itemStyle: {borderColor: '#fff', borderWidth: 2}}];
    } else {
      option.xAxis = {type: spec.type === 'scatter' ? 'value' : 'category', name: spec.x_label,
        nameLocation: 'middle', nameGap: 50, nameTextStyle: {fontSize: 11}, scale: spec.type === 'scatter',
        data: spec.type === 'scatter' ? undefined : (spec.type === 'box' ? [spec.x_label] : spec.labels),
        axisLabel: {hideOverlap: true, fontSize: 10}};
      option.yAxis = {type: 'value', name: spec.y_label, nameGap: 20, scale: spec.type === 'scatter',
        splitLine: {lineStyle: {color: '#edf1ec'}}, axisLabel: {fontSize: 10}};
      if (spec.type === 'scatter') {
        option.legend = {top: 47, data: spec.trend ? ['Records', 'Linear trend'] : ['Records'], textStyle: {fontSize: 10}};
        option.series = [{name: 'Records', type: 'scatter', data: spec.points, symbolSize: 8,
          itemStyle: {opacity: .65}, emphasis: {itemStyle: {opacity: 1}}}];
        if (spec.groups?.length) {
          option.legend.data = spec.groups.map(group => group.name);
          option.series = spec.groups.map(group => ({name: group.name, type: 'scatter', data: group.points,
            symbolSize: 8, itemStyle: {opacity: .7}}));
        }
        if (spec.trend) option.series.push({name: 'Linear trend', type: 'line', data: spec.trend,
          showSymbol: false, lineStyle: {color: '#c29a57', type: 'dashed', width: 2}});
        option.dataZoom = [{type: 'inside'}, {type: 'slider', bottom: 9, height: 15}];
      } else if (spec.type === 'box') {
        option.yAxis.name = spec.x_label;
        option.xAxis.name = '';
        option.series = [{type: 'boxplot', data: [spec.values]}];
      } else {
        option.series = [{type: spec.type === 'line' ? 'line' : 'bar', data: spec.values,
          barMaxWidth: 55, itemStyle: {borderRadius: [4, 4, 0, 0]}, showSymbol: spec.type === 'line'}];
        if ((spec.labels || []).length > 10) option.dataZoom = [{type: 'inside'}, {type: 'slider', bottom: 9, height: 15}];
      }
    }
    chart.setOption(option);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(container);
    container.chartInstance = chart;
    container.chartObserver = observer;
  }
};
