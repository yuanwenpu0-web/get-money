import { init, use } from 'echarts/core'
import { BarChart, CandlestickChart, LineChart } from 'echarts/charts'
import { AxisPointerComponent, DataZoomComponent, GridComponent, LegendComponent, MarkPointComponent, TooltipComponent } from 'echarts/components'
import { LabelLayout } from 'echarts/features'
import { CanvasRenderer } from 'echarts/renderers'

use([BarChart, CandlestickChart, LineChart, AxisPointerComponent, DataZoomComponent, GridComponent, LegendComponent, MarkPointComponent, TooltipComponent, LabelLayout, CanvasRenderer])

export { init }
