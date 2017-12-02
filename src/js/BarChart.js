import Svg from './Svg';
import React from 'react';
import '../css/bar_charts.css';

const d3 = window.d3;

export default class Bar extends React.Component {
    getBarData() {
        let set = (this.props.topic === 1)
            ? this.props.topics[this.props.topicA].words.slice(0, 20)
            : this.props.topics[this.props.topicB].words.slice(0, 20)
        set = set.map(v => {
            return {
                p_topic: v.prob,
                count: v.count,
                name: v.name
            }});       
        return set;
    }

    getColor() {
        const tColor = (this.props.topic === 1) 
            ? this.props.topicA_color 
            : this.props.topicB_color
        return tColor;
    }

    render() {
        return (
            <div className="bar-container">
                <BarChart 
                    data={this.getBarData()} 
                    color={this.getColor()}
                />
            </div>
        );
    }
}

const BarChart = Svg((node, props) => {  

    // chart dimensions
    var width = 300,
        height = 420,
        barWidth = 20,
        barSpacing = 1;

    const dataMax = d3.max(props.data.map(v => {
        return v.p_topic
    }));

    function prob(d) { return d.p_topic; }    

    const lengthScale = d3.scaleLinear()
        .domain([0, dataMax])
        .range([0, width-80])
    
    const colorScale = d3.scaleLinear()
        .domain([0, dataMax])
        .range(['#444', props.color])

    var svg = d3.select(node);
    svg.selectAll('*').remove();

    svg.attr('height', height)

    // add words
    svg.selectAll('g')
    .data(props.data)
    .enter()
    .append('text')        
        .attr('y', (d, i) => i * (barWidth + barSpacing))
        .attr('dy', '1em')
        .attr('x', d => 0)
        .attr('class', 'bar-title')
        .text(function(d) { return d.name; })   

    // add bars
    svg.selectAll('rect')
        .data(props.data)
        .enter()
        .append('rect')
            .attr('y', (d, i) => i * (barWidth + barSpacing))
            .attr('x', d => 100)
            .attr('width', d => lengthScale(prob(d)))
            .attr('height', barWidth)
            .style('fill', d => colorScale(prob(d)))
            .on('mouseover', (data, index, nodes) => {
                d3.select(nodes[index])
                    .style('fill', '#444')        

                d3.select("#tooltip")    
                    .html(
                        "<h4>" + data.name + "</h4>" +
                        "<p><em>probability:</em> " + data.p_topic +
                        "</br><em>count:</em> " + data.count +
                        "</p>"
                    )
                    
            })
            .on('mouseout', function (data, index, nodes) {
                d3.select(nodes[index])
                    .transition()
                    .duration(200)
                    .style('fill', d => colorScale(prob(d)));

                d3.select("#tooltip")
                    .attr('style',
                        'opacity:0;border: 1px solid ' + props.color +
                        ';top:' + (d3.event.clientY - 10) + 
                        'px;left:' + (d3.event.clientX + 10) + "px")      
            })
            .on('mousemove', function(data, index, nodes) {                
                d3.select("#tooltip")
                    .attr('style', 
                        'opacity:.9;border: 1px solid ' + props.color +
                        ';top:' + (d3.event.clientY - 10) + 
                        'px;left:' + (d3.event.clientX + 10) + "px")      
            })    
});