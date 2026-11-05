import { init, use } from 'echarts/core'
import { BarChart, CandlestickChart, LineChart, ScatterChart } from 'echarts/charts'
import { AxisPointerComponent, DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, TooltipComponent } from 'echarts/components'
import { LabelLayout } from 'echarts/features'
import { CanvasRenderer } from 'echarts/renderers'

use([BarChart, CandlestickChart, LineChart, ScatterChart, AxisPointerComponent, DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, TooltipComponent, LabelLayout, CanvasRenderer])

export { init }
