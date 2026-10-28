import { init, use } from 'echarts/core'
import { BarChart, CandlestickChart, LineChart } from 'echarts/charts'
import { AxisPointerComponent, DataZoomComponent, GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([BarChart, CandlestickChart, LineChart, AxisPointerComponent, DataZoomComponent, GridComponent, TooltipComponent, CanvasRenderer])

export { init }
