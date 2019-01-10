import Svg from './Svg';
import React from 'react';
import '../css/topic_bar.css';

const d3 = window.d3;

export default class TopicBarChart extends React.Component {
    render() {
        return (
            <div className="topic-bar-container">
                <Graph
                    data={this.props.topicScores}
                    topicA={this.props.topicA}
                    topicB={this.props.topicB}
                    colorA={this.props.topicA_color}
                    colorB={this.props.topicB_color}
                    topicFifo={this.props.topicFifo}
                />
            </div>
        );
    }
}

const Graph = Svg((node, props) => {
    const scores = props.data.map(i => {
        return i[0]
    });
    const counts = props.data.map(i => {
        return i[1]
    })
    
    // chart dimensions
    var margin = { top: 40, right: 10, bottom: 20, left: 0 },
        width = (1000 - margin.left - margin.right) / 2 + 160,
        height = 300 - margin.top - margin.bottom,
        barWidth = 10.8,
        barSpacing = 1;

    const dataMax = d3.max(scores);
    function prob(d) { return d; }

    // position scales
    const lengthScale = d3.scaleLinear()
        .domain([0, dataMax])
        .range([0, height])

    // coloring
    const colorScale = d3.scaleLinear()
        .domain([0, 1])
        .range([props.colorA, props.colorB])

    const color = (d, i) => {
        if (i === props.topicA) return colorScale(0)
        if (i === props.topicB) return colorScale(1)
        return "#394B59"
    }

    // axes    
    var x = d3.scaleLinear()
        .domain([0, 50])
        .range([0, width - 72]);

    var y = d3.scaleLinear()
        .domain([0, dataMax])
        .range([height, 0]);

    var xAxis = d3.axisBottom(x)
        .tickSize(10)
        .tickPadding(6);

    var yAxis = d3.axisLeft(y)
        .tickSize(10)
        .tickPadding(6);

    // build the vis
    var svg = d3.select(node);
    svg.selectAll('*').remove();

    svg.attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom + 40)
        .style("padding", "0px 0px 0px 0px")

    // add bars
    svg.selectAll('rect')
        .data(scores)
        .enter()
        .append('rect')
        .attr('x', (d, i) => i * (barWidth + barSpacing))
        .attr('height', d => lengthScale(prob(d)))
        .attr('y', d => height - lengthScale(prob(d)))
        .attr('width', barWidth)
        .style('fill', (d, i) => color(d, i))
        .on('mouseover', (data, index, nodes) => {
            d3.select(nodes[index])
                .style('fill', "#5C7080")
            d3.select("#tooltip")
                .html("<h4>Topic " + (index+1) + "</h4>" +
                    "<p><em>average score:</em> " + data + "</br>" +
                    "<em>number of jokes:<em> " + counts[index] +
                    "</p>")            
        })
        .on('mousemove', function (data, index, nodes) {
            let xPos = d3.event.clientX + 10;
            if (xPos > 1000) xPos -= 220

            let col = color(data, index);
            if (col === "#394B59")
                col = "#5C7080"

            d3.select("#tooltip")
                .attr('style',
                'opacity:.95;border: 1px solid ' + col +
                ';border-top: 15px solid ' + col +
                ';top:' + (d3.event.clientY + 10) +
                'px;height:100px;' +
                'left:' + xPos + "px")

        })
        .on('mouseout', function (data, index, nodes) {
            let col = color(data, index);
            if (col === "#394B59")
                col = "#5C7080"

            let xPos = d3.event.clientX + 10;
            if (xPos > 1000) xPos -= 220

            d3.select(nodes[index])
                .transition()
                .duration(200)
                .style('fill', color(data, index));

            d3.select("#tooltip")
                .attr('style',
                'opacity:0;border: 1px solid ' + col +
                ';border-top: 15px solid ' + col +
                ';top:' + (d3.event.clientY + 10) +
                'px;height:100px;' +
                'px;left:' + xPos + "px")
        })
        .on('click', function (data, index, nodes) {
            props.topicFifo(index)
        })

    // the red dotted line
    const total_mean = d3.mean(scores)
    svg.append("line")
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "5 5")
        .attr("x1", 0)
        .attr("y1", height - lengthScale(total_mean))
        .attr("x2", width - 64)
        .attr("y2", height - lengthScale(total_mean))
    // add the label
    svg.append("text")
        .attr("text-anchor", 'left')
        .attr("class", "topic_label")
        .attr("x", 5)
        .attr("y", (height - lengthScale(total_mean) - 5))
        .text(total_mean.toFixed(3))

    // add the x-axis
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    // text label for the x axis
    svg.append("text")
        .attr("transform",
        "translate(" + (width / 2) + " ," +
        (height + margin.top) + ")")
        .style("text-anchor", "middle")
        .text("Topic Number");

    // add the y-axis
    svg.append("g")
        .attr("class", "y axis")
        .attr("transform", "translate(0," + -0.5 + ")")
        .call(yAxis);

    // text label for the y axis
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left)
        .attr("x", 0 - (height / 2))
        .attr("dy", "-2.5em")
        .style("text-anchor", "middle")
        .text("Average Joke Score");

    // add a graph title
    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .text("Average scores per topic");
});